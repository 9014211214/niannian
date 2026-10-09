"""Explicit documentation boundary; all cases are software policy checks."""
import hashlib
import json
import zipfile

import pytest

from scripts import build_minimal_release as release


def test_policy14_scope_is_exactly_public_aggregate_and_preparation_documents():
    assert release.V14_POLICY_VERSION == "aria-minimal-source-14"
    assert release.V14_POLICY_FILES == {
        "docs/E1CV2_RESULTS_2026_10_07.md", "docs/E1C_REPAIR_2026_10_07.md",
        "docs/VALIDATION_PREPARATION_2026_10_07.md", "docs/THREE_SESSION_ACCEPTANCE_2026_10_07.md",
        "docs/VALIDATION_RESULTS_2026_10_07.md",
    }
    assert all(release._allowed(p) for p in release.V14_POLICY_FILES)
    assert all(not release._allowed(p, policy) for p in release.V14_POLICY_FILES
               for policy in release.SUPPORTED_POLICIES - {release.V14_POLICY_VERSION, release.POLICY_VERSION})
    for name in ("artifacts/local/tasks/cases.json", "artifacts/local/first-labels.json",
                 "docs/private_answers.md", ".env", "data/persona.json", "docs/media/demo.mp4"):
        assert not release._allowed(name)


def test_v13_original_scope_remains_verifiable(tmp_path):
    root=tmp_path/'source'
    for name in release.REQUIRED_FILES:
        path=root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('# authored software fixture\n',encoding='utf-8')
    target=tmp_path/'new.zip'
    release.build_release(root,target)
    with zipfile.ZipFile(target) as z:
        data={n:z.read(n) for n in z.namelist()}
    manifest=json.loads(data['RELEASE_MANIFEST.json'])
    manifest['policy_version']=release.V13_POLICY_VERSION
    data['RELEASE_MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()+b'\n'
    data['RELEASE_SCOPE.md']=release.V13_SCOPE_TEXT.encode()
    old=tmp_path/'old.zip'
    with zipfile.ZipFile(old,'w') as z:
        for n,raw in data.items():z.writestr(n,raw)
    assert release.verify_release(old)['policy_version']==release.V13_POLICY_VERSION
    illegal=tmp_path/'illegal.zip'
    name='docs/VALIDATION_RESULTS_2026_10_07.md'
    raw=b'# synthetic aggregate\n'
    manifest['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    data[name]=raw
    data['RELEASE_MANIFEST.json']=json.dumps(manifest,ensure_ascii=False,indent=2).encode()+b'\n'
    with zipfile.ZipFile(illegal,'w') as z:
        for n,raw in data.items():z.writestr(n,raw)
    with pytest.raises(release.ReleaseError):release.verify_release(illegal)
