const $ = id => document.getElementById(id);
let selectedDoc = null, selectedSelfie = null;

function classificationClass(c){
  return c === "HIGH_RISK" ? "risk-high" : c === "REVIEW_REQUIRED" ? "risk-review" : "risk-low";
}
function showError(msg){ $("error").textContent=msg; $("error").classList.remove("hidden"); }
function clearError(){ $("error").classList.add("hidden"); }
function previewFile(file){
  $("preview").innerHTML="";
  if(file && file.type.startsWith("image/")){
    const img=document.createElement("img"); img.src=URL.createObjectURL(file); $("preview").appendChild(img);
  } else if(file){ $("preview").innerHTML=`<span class="muted">${file.name}</span>`; }
}
$("document").addEventListener("change", e=>{
  selectedDoc=e.target.files[0]||null;
  if(selectedDoc){ $("drop-title").textContent=selectedDoc.name; $("drop-sub").textContent="Document selected"; previewFile(selectedDoc); }
});
$("selfie").addEventListener("change", e=>{selectedSelfie=e.target.files[0]||null;});
$("reset").addEventListener("click",()=>{
  selectedDoc=null; selectedSelfie=null; $("document").value=""; $("selfie").value="";
  $("drop-title").textContent="Drop identity document here"; $("drop-sub").textContent="JPG, PNG or PDF · max 10 MB";
  $("preview").innerHTML=""; $("result").classList.add("hidden"); $("empty").classList.remove("hidden"); clearError();
});
$("analyze").addEventListener("click", async ()=>{
  clearError();
  if(!selectedDoc){showError("Please choose an identity document.");return;}
  $("loading").classList.remove("hidden"); $("analyze").disabled=true;
  const fd=new FormData(); fd.append("document",selectedDoc); if(selectedSelfie)fd.append("selfie",selectedSelfie);
  try{
    const res=await fetch("/api/v1/verify/document",{method:"POST",body:fd});
    const data=await res.json();
    if(!res.ok) throw new Error(data.detail||"Analysis failed.");
    renderResult(data); loadStats(); loadHistory();
  }catch(err){showError(err.message);}
  finally{$("loading").classList.add("hidden");$("analyze").disabled=false;}
});
function renderResult(r){
  $("empty").classList.add("hidden"); $("result").classList.remove("hidden");
  $("score").textContent=r.risk_score; $("classification").textContent=r.classification.replaceAll("_"," ");
  $("classification").className=classificationClass(r.classification);
  $("request").textContent=`Request ${r.request_id}`;
  $("result-tag").textContent="COMPLETED";
  const face=r.face_verification;
  $("checks").innerHTML=`
    <div class="check"><small>OCR CONFIDENCE</small><b>${r.document.ocr_confidence==null?"N/A":(r.document.ocr_confidence*100).toFixed(1)+"%"}</b></div>
    <div class="check"><small>ANOMALY SCORE</small><b>${((r.explanations.anomaly_score||0)*100).toFixed(1)}%</b></div>
    <div class="check"><small>FACE VERIFICATION</small><b>${face.performed?(face.similarity*100).toFixed(1)+"% similarity":"NOT PERFORMED"}</b></div>
    <div class="check"><small>DETECTED FLAGS</small><b>${r.anomalies.length}</b></div>`;
  const reasons=r.explanations.reasons||[];
  $("reasons").innerHTML=reasons.length?reasons.map(x=>`<li>${escapeHtml(x)}</li>`).join(""):"<li>No major risk reason was generated.</li>";
  $("fields").textContent=JSON.stringify(r.document.extracted_fields,null,2);
}
async function loadStats(){
  try{const r=await fetch("/api/v1/dashboard/stats").then(x=>x.json());$("total").textContent=r.total;$("low").textContent=r.low_risk;$("review").textContent=r.review_required;$("high").textContent=r.high_risk;}catch{}
}
async function loadHistory(){
  try{
    const rows=await fetch("/api/v1/screenings").then(x=>x.json());
    $("history").innerHTML=rows.length?rows.map(r=>`<tr><td>${r.request_id.slice(0,8)}…</td><td>${escapeHtml(r.filename)}</td><td><b>${r.risk_score}</b></td><td class="${classificationClass(r.classification)}">${r.classification.replaceAll("_"," ")}</td><td>${new Date(r.created_at).toLocaleString()}</td></tr>`).join(""):'<tr><td colspan="5" class="muted">No screenings yet.</td></tr>';
  }catch{}
}
function escapeHtml(s){return String(s).replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]));}
$("refresh").addEventListener("click",()=>{loadStats();loadHistory();});
loadStats(); loadHistory();
