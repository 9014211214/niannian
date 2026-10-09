"""Read and summarize the bound historical evidence. Never rerun its tests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
MANIFEST_SHA = '30ec6d9b152999ed51b5cdf3ffb33e568f6484534b4043917ef16274763b9ec2'
REPOSITORY = '9014211214/niannian'
REPOSITORY_ID = 1404555679
SOURCE_COMMIT = '461dc4abfde329b73c26b036cd2f8a2c2dd6bd73'
SOURCE_RUN = 37651687095
SOURCE_ATTEMPT = 1
API = 'https://api.github.com/repos/'+REPOSITORY
API_LIMIT = 1024*1024


class Rejected(ValueError):
    """Only fixed codes may escape the boundary; URLs and credentials never do."""


def require(condition, code):
    if not condition:
        raise Rejected(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_binding(directory=HERE):
    raw = (directory/'recovery-manifest.json').read_bytes()
    require(sha(raw) == MANIFEST_SHA, 'frozen_manifest_changed')
    manifest = json.loads(raw)
    require((manifest['repository'],manifest['repository_id'],manifest['source_commit'],
             manifest['source_run_id'],manifest['source_run_attempt']) ==
            (REPOSITORY,REPOSITORY_ID,SOURCE_COMMIT,SOURCE_RUN,SOURCE_ATTEMPT),
            'frozen_source_binding_changed')
    require(len(manifest['artifacts']) == 16 and
            len({a['id'] for a in manifest['artifacts']}) == 16,
            'frozen_artifact_set_changed')
    require(sha((directory/'frozen_ci_tests.py').read_bytes()) ==
            manifest['frozen_summary_tool_sha256'], 'frozen_summary_source_changed')
    return manifest


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


class HTTPReadOnly:
    """Two independent GETs; never redirect the authenticated API request."""
    def __init__(self):
        self.opener = urllib.request.build_opener(NoRedirect())

    def get(self, url, headers, maximum):
        request = urllib.request.Request(url, headers=headers, method='GET')
        try:
            response = self.opener.open(request, timeout=60)
        except urllib.error.HTTPError as error:
            with error:
                # A signed location is inspected by the caller, never printed.
                return error.code, dict(error.headers), b''
        except (OSError, urllib.error.URLError):
            raise Rejected('http_transport_failed') from None
        with response:
            data = response.read(maximum+1)
            require(len(data) <= maximum, 'http_body_limit_exceeded')
            return response.status, dict(response.headers), data


def storage_url(location):
    try:
        url = urllib.parse.urlsplit(location)
        host = (url.hostname or '').lower()
        approved = (host == 'results-receiver.actions.githubusercontent.com' or
                    (host.endswith('.blob.core.windows.net') and
                     host != 'blob.core.windows.net'))
        require(url.scheme == 'https' and approved and not url.username and
                not url.password and url.port in (None,443) and not url.fragment,
                'unapproved_artifact_storage_url')
    except ValueError:
        raise Rejected('unapproved_artifact_storage_url') from None
    return location


class OriginalArtifactReader:
    def __init__(self, manifest, token, transport=None):
        require(bool(token), 'github_actions_read_token_required')
        self.manifest = manifest
        self.token = token
        self.transport = transport or HTTPReadOnly()
        self.requests = []

    def api(self, suffix, *, redirect=False):
        # All suffixes are generated from bound integer IDs, never user input.
        require(suffix.startswith('/') and '?' not in suffix and '..' not in suffix,
                'invalid_api_suffix')
        headers = {'Accept':'application/vnd.github+json',
                   'Authorization':'Bearer '+self.token,
                   'X-GitHub-Api-Version':'2026-03-10',
                   'User-Agent':'aria-bound-original-evidence-reader-v2'}
        status, response_headers, raw = self.transport.get(API+suffix,headers,API_LIMIT)
        self.requests.append({'endpoint':suffix,'http_status':status})
        if redirect:
            require(status == 302, 'artifact_download_api_not_redirect')
            lower = {key.lower():value for key,value in response_headers.items()}
            require('location' in lower, 'artifact_download_location_missing')
            return storage_url(lower['location'])
        require(status == 200, 'github_api_read_failed')
        try:
            return json.loads(raw)
        except (ValueError,UnicodeError):
            raise Rejected('github_api_invalid_json') from None

    def check_attempt(self):
        record = self.api('/actions/runs/'+str(SOURCE_RUN)+'/attempts/1')
        require((record.get('id'),record.get('head_sha'),record.get('run_attempt')) ==
                (SOURCE_RUN,SOURCE_COMMIT,1), 'original_attempt_identity_mismatch')
        repository = record.get('repository',{})
        require(repository.get('id') == REPOSITORY_ID and
                repository.get('full_name') == REPOSITORY,
                'original_repository_identity_mismatch')
        return {'run_id':SOURCE_RUN,'run_attempt':1,'source_commit':SOURCE_COMMIT,
                'observed_status':record.get('status'),
                'observed_conclusion':record.get('conclusion')}

    def read(self, artifact):
        identifier = str(artifact['id'])
        meta = self.api('/actions/artifacts/'+identifier)
        run = meta.get('workflow_run',{})
        require((meta.get('id'),meta.get('name'),meta.get('size_in_bytes'),meta.get('digest')) ==
                (artifact['id'],artifact['name'],artifact['bytes'],
                 'sha256:'+artifact['zip_sha256']), 'original_artifact_metadata_mismatch')
        require(meta.get('expired') is False, 'original_artifact_expired')
        require((run.get('id'),run.get('head_sha'),run.get('repository_id'),
                 run.get('head_repository_id')) ==
                (SOURCE_RUN,SOURCE_COMMIT,REPOSITORY_ID,REPOSITORY_ID),
                'original_artifact_run_identity_mismatch')
        location = self.api('/actions/artifacts/'+identifier+'/zip',redirect=True)
        # No API Authorization, cookies, or API headers enter signed storage.
        status, headers, raw = self.transport.get(location,{},artifact['bytes'])
        require(status == 200, 'artifact_storage_read_failed')
        require(len(raw) == artifact['bytes'] and sha(raw) == artifact['zip_sha256'],
                'original_zip_identity_mismatch')
        return raw


def validate_zip(raw, record, manifest):
    require(len(raw) == record['bytes'] and sha(raw) == record['zip_sha256'],
            'original_zip_identity_mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as packed:
        all_infos = packed.infolist()
        names = [item.filename for item in all_infos]
        require(len(names) == len(set(names)), 'duplicate_zip_member')
        for item in all_infos:
            path = PurePosixPath(item.filename)
            require(not path.is_absolute() and '..' not in path.parts and
                    '\\' not in item.filename and ':' not in item.filename and
                    not stat.S_ISLNK(item.external_attr >> 16), 'unsafe_zip_member')
        infos = [item for item in all_infos if not item.is_dir()]
        expected = {item['path']:item for item in record['files']}
        require({item.filename for item in infos} == set(expected),
                'original_member_set_changed')
        contents = {}
        for item in infos:
            target = expected[item.filename]
            require(item.file_size == target['bytes'], 'original_member_identity_changed')
            data = packed.read(item)
            require(len(data) == target['bytes'] and sha(data) == target['sha256'],
                    'original_member_identity_changed')
            contents[item.filename] = data
    context = json.loads(contents['context.json'])
    require((context.get('source_commit'),str(context.get('run_id')),
             str(context.get('run_attempt'))) ==
            (SOURCE_COMMIT,str(SOURCE_RUN),'1'), 'source_commit_run_attempt_changed')
    return contents


def execute(manifest, output, *, archives=None, reader=None, execution_identity=None):
    require(not output.exists(), 'fresh_output_required')
    require((archives is None) != (reader is None), 'exactly_one_evidence_source_required')
    require(output.resolve() != HERE.resolve() and HERE.resolve() not in output.resolve().parents,
            'output_must_be_outside_candidate_tools')
    observed_attempt = reader.check_attempt() if reader else None
    output.mkdir(parents=True,exist_ok=False)
    inputs = output/'private'/'inputs'
    inputs.mkdir(parents=True)
    public = output/'public'
    public.mkdir()
    bindings = []
    for record in manifest['artifacts']:
        raw = reader.read(record) if reader else (archives/('artifact-'+str(record['id'])+'.zip')).read_bytes()
        contents = validate_zip(raw,record,manifest)
        for name,data in contents.items():
            target = inputs/('artifact-'+str(record['id']))/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
        bindings.append({'artifact_id':record['id'],'name':record['name'],
                         'sha256':sha(raw),'bytes':len(raw)})
    environment = {name:os.environ[name] for name in
                   ('PATH','SystemRoot','WINDIR','COMSPEC','PATHEXT','TEMP','TMP')
                   if name in os.environ}
    # Only the child frozen summary sees old-run identity. The workflow retains
    # its own run/attempt; no old artifact or context is edited.
    environment.update(GITHUB_RUN_ID=str(SOURCE_RUN),GITHUB_RUN_ATTEMPT='1',PYTHONUTF8='1')
    summary = public/'four-platforms.original-evidence.json'
    process = subprocess.run([sys.executable,'-X','utf8',str(HERE/'frozen_ci_tests.py'),
                              'summary','--directory',str(inputs),'--output',str(summary),
                              '--commit',SOURCE_COMMIT],env=environment,
                             capture_output=True,timeout=60)
    require(process.returncode == 0, 'frozen_original_summary_rejected')
    result = json.loads(summary.read_bytes())
    require((result.get('status'),result.get('source_commit'),result.get('run_id'),
             result.get('run_attempt')) == ('pass',SOURCE_COMMIT,str(SOURCE_RUN),'1'),
            'unexpected_original_summary_result')
    identity = execution_identity or {'kind':'local_offline_fixture'}
    receipt = {'schema':'aria_cross_attempt_original_summary_recovery_v2',
               'at_utc':datetime.now(timezone.utc).isoformat(),
               'status':'bound_original_evidence_summary_pass',
               'repository':REPOSITORY,'repository_id':REPOSITORY_ID,
               'source_commit':SOURCE_COMMIT,'source_run_id':SOURCE_RUN,'source_run_attempt':1,
               'recovery_execution':identity,'original_attempt_observation':observed_attempt,
               'manifest_sha256':MANIFEST_SHA,
               'frozen_summary_sha256':manifest['frozen_summary_tool_sha256'],
               'recovery_tool_sha256':sha(Path(__file__).read_bytes()),
               'summary_sha256':sha(summary.read_bytes()),'artifacts':bindings,
               'original_remote_conclusion_remains':'failure',
               'current_product_qualified':False,'model_effect_claim':False,
               'software_tests_rerun':False,'historical_contexts_modified':False,
               'remote_mutating_api_calls':0,
               'network_mode':'read_only_get' if reader else 'offline',
               'raw_artifacts_in_public_output':False}
    (public/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    if reader:
        (output/'private'/'api-read-receipt.json').write_text(
            json.dumps(reader.requests,indent=2)+'\n',encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--archives',type=Path)
    choice.add_argument('--download',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try:
        manifest = load_binding()
        if args.download:
            require(os.environ.get('GITHUB_REPOSITORY') == REPOSITORY,
                    'recovery_execution_repository_mismatch')
            require(os.environ.get('GITHUB_EVENT_NAME') == 'workflow_dispatch',
                    'manual_recovery_execution_required')
            run = os.environ.get('GITHUB_RUN_ID','')
            attempt = os.environ.get('GITHUB_RUN_ATTEMPT','')
            commit = os.environ.get('GITHUB_SHA','')
            require(run.isdecimal() and attempt.isdecimal() and len(commit) == 40 and
                    all(c in '0123456789abcdef' for c in commit),
                    'recovery_execution_identity_missing')
            identity = {'kind':'new_manual_workflow','run_id':int(run),
                        'run_attempt':int(attempt),'source_commit':commit}
            reader = OriginalArtifactReader(manifest,os.environ.get('GITHUB_TOKEN',''))
            receipt = execute(manifest,args.output,reader=reader,execution_identity=identity)
        else:
            receipt = execute(manifest,args.output,archives=args.archives)
        print(json.dumps({'status':receipt['status'],'source_run_id':SOURCE_RUN,
                          'source_run_attempt':1,'artifacts':len(receipt['artifacts']),
                          'current_product_qualified':False}))
        return 0
    except Rejected as error:
        print(json.dumps({'status':'rejected','code':str(error)}),file=sys.stderr)
    except (OSError,ValueError,KeyError,zipfile.BadZipFile,subprocess.TimeoutExpired):
        print(json.dumps({'status':'rejected','code':'evidence_or_execution_invalid'}),file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
