"""Run actual shipped stream/replay JavaScript on inert saved-frame fixtures."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def run_js(scenario, version="task_response_v3"):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    html = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf8")
    functions = html[html.index("    async function requestTextStream(body)"):html.index("    async function refresh()")]
    replay = html[html.index("    function framesFor(turns)"):html.index("    function renderStage()")]
    script = r'''
const scenario=SCENARIO;
const version=scenario==='legacy'?'task_response_v1':VERSION;
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
if(scenario.startsWith('waiting')) {
  const waiting={...body,status:'waiting',request_sent:false};
  if(scenario!=='waiting_plain')waiting.protection_prelude=structuredClone(prelude);
  if(scenario==='waiting_changed_text')waiting.text='changed';
  const waitingFrame={type:'input_waiting',turn_id:'t',session_id:'s',state:{session_id:'s',premodel_safety_version:version,turns:[],waiting_input:waiting}};
  lines=scenario==='waiting_plain'?[waitingFrame,...lines]:[waitingFrame,lines[1]];
  if(scenario==='waiting_changed_prelude')final.turns[0].protection_prelude.protection='changed';
  if(scenario==='waiting_duplicate')lines.unshift(waitingFrame);
} 
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
'''.replace("SCENARIO", json.dumps(scenario)).replace("VERSION", json.dumps(version)).replace("FUNCTIONS", functions).replace("REPLAY", replay)
    result = subprocess.run([node, "-"], input=script, encoding="utf8", capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("version", ["task_response_v2", "task_response_v3", "task_response_v4"])
@pytest.mark.parametrize("scenario", ["complete", "legacy"])
def test_actual_stream_accepts_exact_saved_prelude_and_final_version(scenario, version):
    result = run_js(scenario, version)
    assert result["accepted"] and result["renders"] == 2
    assert result["paths"] == ["/api/minimal/text-stream"]
    if scenario == "complete":
        assert [row["text"] for row in result["frames"]] == ["neutral", "old protection", "new body"]
    else:
        assert len(result["frames"]) == 2


@pytest.mark.parametrize("version", ["task_response_v2", "task_response_v3", "task_response_v4"])
@pytest.mark.parametrize("scenario", ["changed_prelude", "changed_version", "missing_prelude", "changed_input", "duplicate_row", "missing_final"])
def test_actual_stream_rejects_drift_and_keeps_first_visible_protection(scenario, version):
    result = run_js(scenario, version)
    assert not result["accepted"] and result["renders"] == 1
    assert result["last"]["turns"][0]["protection_prelude"]["protection"] == "old protection"



@pytest.mark.parametrize("scenario", ["waiting_plain", "waiting_protection"])
def test_actual_default_stream_accepts_saved_wait_and_displays_protection_once(scenario):
    result = run_js(scenario, "task_response_v4")
    assert result["accepted"]
    assert result["renders"] == (3 if scenario == "waiting_plain" else 2)
    assert [row["text"] for row in result["frames"]] == ["neutral", "old protection", "new body"]


@pytest.mark.parametrize("scenario", ["waiting_changed_prelude", "waiting_duplicate", "waiting_changed_text"])
def test_actual_default_stream_rejects_waiting_identity_or_protection_drift(scenario):
    result = run_js(scenario, "task_response_v4")
    assert not result["accepted"]
    assert result["renders"] == (0 if scenario == "waiting_changed_text" else 1)



@pytest.mark.parametrize("version", ["task_response_v2", "task_response_v3", "task_response_v4"])
def test_actual_conversation_render_preserves_separate_prelude_once(version):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    html = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf8")
    function = html[html.index("    function renderConversation()"):html.index("    function evidenceCard(")]
    script = r'''
const output=[];
const element=()=>({append:()=>{},replaceChildren:()=>{},setAttribute:()=>{}});
const document={createElement:element}, nodes=new Map(), $=id=>{if(!nodes.has(id))nodes.set(id,element());return nodes.get(id)};
const array=v=>Array.isArray(v)?v:[],scalar=v=>v||'',label=v=>v||'',displayTime=v=>v||'',shortId=v=>v;
const viewingHistory=()=>false,isCurrentOpen=()=>true,renderStage=()=>{},fillInputEvidence=()=>{},fields=()=>{},placeholder=()=>{},text=()=>element();
const message=(role,body,source)=>{output.push({role,body,source});return element()};
let selectedSession='s',activeSession='s',replayMode=false,renderingTurns='';
const state={mode:'live',session_id:'s',sessions:[{session_id:'s',number:1}],turns:[{session_id:'s',turn_id:'t',text:'input',status:'completed',response:'body',source:'live',protection_prelude:{premodel_safety_version:VERSION,protection:'protection'}}]};
FUNCTION
renderConversation();console.log(JSON.stringify(output));
'''.replace("VERSION", json.dumps(version)).replace("FUNCTION", function)
    result = subprocess.run([node, "-"], input=script, encoding="utf8", capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert [row["body"] for row in json.loads(result.stdout)] == ["input", "protection", "body"]


@pytest.mark.parametrize("endpoint,body,accepted", [
    ("preference", {"action": "withdraw"}, True),
    ("cognitive", {"action": "delete"}, True),
    ("seal", {"session_id": "s"}, True),
    ("forget", {}, True),
    ("text", {"turn_id": "another", "text": "new"}, False),
    ("preference", {"action": "confirm"}, False),
])
def test_actual_waiting_ui_keeps_owner_reductions_available_without_second_turn(endpoint, body, accepted):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    html = (Path(__file__).parents[1] / "aria/assets/voice/minimal.html").read_text(encoding="utf8")
    function = html[html.index("    async function mutate("):html.index("    $('composer').addEventListener")]
    script = r'''
let busy=true,careBusy=false,replayMode=false,uncertain=null;
let state={session_id:'s',waiting_input:{status:'waiting'},sessions:[]},paths=[];
const array=v=>v||[],viewingHistory=()=>false,modelCallsStopped=()=>false,renderModelUsage=()=>{},status=()=>{},setError=()=>{},updateControls=()=>{},render=()=>{};
const request=async(endpoint,body)=>{paths.push(endpoint);return {ok:true}};
FUNCTION
(async()=>{await mutate(ENDPOINT,BODY);console.log(JSON.stringify({busy,careBusy,paths}));})();
'''.replace("FUNCTION", function).replace("ENDPOINT", json.dumps(endpoint)).replace("BODY", json.dumps(body))
    result = subprocess.run([node, "-"], input=script, encoding="utf8", capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual == {"busy": True, "careBusy": False, "paths": [endpoint] if accepted else []}
