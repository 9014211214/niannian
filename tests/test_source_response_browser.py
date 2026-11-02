"""Run actual shipped stream/replay JavaScript on inert saved-frame fixtures."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def run_js(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    html = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf8")
    functions = html[html.index("    async function requestTextStream(body)"):html.index("    async function refresh()")]
    replay = html[html.index("    function framesFor(turns)"):html.index("    function renderStage()")]
    script = r'''
const scenario=SCENARIO;
const version=scenario==='legacy'?'task_response_v1':'task_response_v2';
let state={premodel_safety_version:version}, renders=[], paths=[];
const TOKEN='software',scalar=v=>typeof v==='string'?v:'',label=v=>v;
const status=()=>{},render=s=>{state=s;renders.push(s)};
const body={session_id:'s',turn_id:'t',text:'neutral'};
const prelude={version:'safety_prelude_v1',premodel_safety_version:version,protection:'old protection',
  protection_sha256:'fixture-hash',user_id:'owner',session_id:'s',scene_id:'companion_elder',turn_index:0,
  utterance_sha256:'fixture-input',saved_at:'time',status:'saved'};
const pending={session_id:'s',premodel_safety_version:version,turns:[{...body,status:'pending',response:'',protection_prelude:prelude}]};
const final=structuredClone(pending);
final.turns[0].status='completed';final.turns[0].response=scenario==='legacy'?'old protection\nbody':'new body';
final.turns[0].protection_prelude.status='transport_returned';
if(scenario==='changed_prelude')final.turns[0].protection_prelude.protection='changed';
if(scenario==='changed_version')final.premodel_safety_version='task_response_v1';
if(scenario==='missing_prelude')delete final.turns[0].protection_prelude;
if(scenario==='changed_input')final.turns[0].text='different';
if(scenario==='duplicate_row')final.turns.push(structuredClone(final.turns[0]));
let lines=[{type:'safety_prelude',turn_id:'t',session_id:'s',state:pending},
 {type:'result',turn_id:'t',session_id:'s',result:{ok:true,state:final}}];
if(scenario==='missing_final')lines.pop();
const data=new TextEncoder().encode(lines.map(v=>JSON.stringify(v)+'\n').join(''));
async function fetch(path){
  paths.push(path);let done=false;
  return {ok:true,headers:{get:()=> 'application/x-ndjson'},body:{getReader:()=>({
    read:async()=>done?{done:true}:(done=true,{done:false,value:data}),
    cancel:async()=>{},releaseLock:()=>{}
  })}};
}
FUNCTIONS
REPLAY
(async()=>{let accepted=false,error='';try{await request('text',body);accepted=true;}catch(e){error=e.message;}
console.log(JSON.stringify({accepted,error,paths,renders:renders.length,last:renders.at(-1),frames:framesFor(final.turns)}));})();
'''.replace("SCENARIO", json.dumps(scenario)).replace("FUNCTIONS", functions).replace("REPLAY", replay)
    result = subprocess.run([node, "-"], input=script, encoding="utf8", capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("scenario", ["complete", "legacy"])
def test_actual_stream_accepts_exact_saved_prelude_and_final_version(scenario):
    result = run_js(scenario)
    assert result["accepted"] and result["renders"] == 2
    assert result["paths"] == ["/api/minimal/text-stream"]
    if scenario == "complete":
        assert [row["text"] for row in result["frames"]] == ["neutral", "old protection", "new body"]
    else:
        assert len(result["frames"]) == 2


@pytest.mark.parametrize("scenario", ["changed_prelude", "changed_version", "missing_prelude", "changed_input", "duplicate_row", "missing_final"])
def test_actual_stream_rejects_drift_and_keeps_first_visible_protection(scenario):
    result = run_js(scenario)
    assert not result["accepted"] and result["renders"] == 1
    assert result["last"]["turns"][0]["protection_prelude"]["protection"] == "old protection"
