/* ==========================================================================
   EDIP — Egypt Development Intelligence Platform
   Client-side Gradient Boosting inference engine (no server, no API calls).
   Reads the trained scikit-learn model exported in data.js (EDIP_DATA.model)
   and re-implements its exact decision-tree ensemble logic in JavaScript.
   ========================================================================== */

const SECTORS = [
  {
    key: "pop", glyph: "١", title: "السكان",
    fields: [
      { id:"Population_Density", label:"الكثافة السكانية", unit:"نسمة/كم²", min:0, max:20000, step:50, decimals:0 },
      { id:"Urban_Percentage",   label:"نسبة التحضر",       unit:"%",       min:0, max:100,  step:1,  decimals:0 },
      { id:"Birth_Rate",         label:"معدل المواليد",      unit:"‰",       min:10, max:35,  step:0.1,decimals:1 },
      { id:"Migration_Rate",     label:"معدل صافي الهجرة",   unit:"%",       min:-10, max:10, step:0.1,decimals:1 },
    ]
  },
  {
    key: "health", glyph: "٢", title: "الصحة",
    fields: [
      { id:"Vaccination_Rate",          label:"معدل التطعيم",         unit:"%", min:60,max:100,step:0.5,decimals:1 },
      { id:"Health_Insurance_Coverage", label:"تغطية التأمين الصحي",  unit:"%", min:0, max:100,step:0.5,decimals:1 },
      { id:"Doctors",                   label:"عدد الأطباء (لكل وحدة)", unit:"طبيب", min:0,max:400,step:1,decimals:0 },
      { id:"Beds",                      label:"عدد الأسرّة",           unit:"سرير", min:0,max:400,step:1,decimals:0 },
    ]
  },
  {
    key: "edu", glyph: "٣", title: "التعليم",
    fields: [
      { id:"Literacy_Rate",         label:"معدل محو الأمية",      unit:"%",  min:50,max:100,step:0.5,decimals:1 },
      { id:"Exam_Score",            label:"متوسط درجات الامتحانات", unit:"%",  min:30,max:100,step:0.5,decimals:1 },
      { id:"Graduation_Rate",       label:"معدل التخرج",          unit:"%",  min:50,max:100,step:0.5,decimals:1 },
      { id:"Dropout_Rate",          label:"معدل التسرب الدراسي",   unit:"%",  min:0, max:25, step:0.1,decimals:1 },
      { id:"Student_Teacher_Ratio", label:"نسبة الطلاب للمعلم",     unit:"طالب/معلم", min:15,max:40,step:0.1,decimals:1 },
    ]
  },
  {
    key: "econ", glyph: "٤", title: "الاقتصاد",
    fields: [
      { id:"GDP_per_Capita",  label:"نصيب الفرد من الناتج المحلي", unit:"جنيه", min:8000,max:56000,step:100,decimals:0 },
      { id:"Employment_Rate", label:"معدل التوظيف",                unit:"%",   min:25,max:60,step:0.5,decimals:1 },
      { id:"Poverty_Rate",    label:"معدل الفقر",                  unit:"%",   min:0, max:55,step:0.5,decimals:1 },
      { id:"Average_Income",  label:"متوسط الدخل الشهري",           unit:"جنيه", min:400,max:3200,step:10,decimals:0 },
    ]
  },
  {
    key: "infra", glyph: "٥", title: "البنية التحتية والخدمات",
    fields: [
      { id:"Electricity_Coverage", label:"تغطية الكهرباء",   unit:"%", min:80,max:100,step:0.5,decimals:1 },
      { id:"Water_Coverage",       label:"تغطية مياه الشرب", unit:"%", min:75,max:100,step:0.5,decimals:1 },
      { id:"Internet_Coverage",    label:"تغطية الإنترنت",   unit:"%", min:30,max:100,step:0.5,decimals:1 },
      { id:"Waste_Collection",     label:"تجميع القمامة",    unit:"%", min:30,max:100,step:0.5,decimals:1 },
      { id:"Water_Quality_Index",  label:"مؤشر جودة المياه", unit:"نقطة", min:25,max:100,step:0.5,decimals:1 },
    ]
  }
];

const FEATURE_IMPORTANCE_ORDER = [
  "Poverty_Rate","Literacy_Rate","GDP_per_Capita","Waste_Collection",
  "Health_Insurance_Coverage","Water_Quality_Index","Exam_Score","Urban_Percentage","Vaccination_Rate"
];
const FEATURE_LABELS = {};
SECTORS.forEach(s => s.fields.forEach(f => FEATURE_LABELS[f.id] = f.label));

/* ---------------------- Gradient Boosting inference --------------------- */

function evalTree(tree, x){
  let node = 0;
  while (tree.left[node] !== -1){
    node = (x[tree.feature[node]] <= tree.threshold[node]) ? tree.left[node] : tree.right[node];
  }
  return tree.value[node];
}

function predictEDPI(featureValues){
  const model = EDIP_DATA.model;
  const x = model.features.map(f => featureValues[f]);
  let pred = model.init_pred;
  for (const tree of model.trees){
    pred += model.learning_rate * evalTree(tree, x);
  }
  return Math.min(1, Math.max(0, pred));
}

/* ------------------------------- UI build -------------------------------- */

function buildForm(){
  const grid = document.getElementById("sectorGrid");
  const stats = EDIP_DATA.stats;

  SECTORS.forEach(sector => {
    const box = document.createElement("div");
    box.className = "sector";

    const h = document.createElement("div");
    h.className = "sector-title";
    h.innerHTML = `<span class="glyph">${sector.glyph}</span><span>${sector.title}</span>`;
    box.appendChild(h);

    sector.fields.forEach(f => {
      const startVal = stats[f.id] ? stats[f.id].median : (f.min + f.max) / 2;

      const wrap = document.createElement("div");
      wrap.className = "field";
      wrap.innerHTML = `
        <div class="field-head">
          <span class="fname">${f.label}</span>
          <span class="fval" id="val_${f.id}">${startVal.toFixed(f.decimals)} ${f.unit}</span>
        </div>
        <input type="range" id="in_${f.id}" min="${f.min}" max="${f.max}" step="${f.step}" value="${startVal}">
      `;
      box.appendChild(wrap);
    });

    grid.appendChild(box);
  });

  // live label updates
  SECTORS.forEach(sector => sector.fields.forEach(f => {
    const input = document.getElementById(`in_${f.id}`);
    const label = document.getElementById(`val_${f.id}`);
    input.addEventListener("input", () => {
      label.textContent = `${parseFloat(input.value).toFixed(f.decimals)} ${f.unit}`;
    });
  }));
}

function buildGovSelect(){
  const sel = document.getElementById("govSelect");
  Object.keys(EDIP_DATA.gov).forEach(name => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    sel.appendChild(opt);
  });
  sel.addEventListener("change", () => {
    if (!sel.value) return;
    const preset = EDIP_DATA.gov[sel.value];
    SECTORS.forEach(sector => sector.fields.forEach(f => {
      if (preset[f.id] === undefined) return;
      const input = document.getElementById(`in_${f.id}`);
      const label = document.getElementById(`val_${f.id}`);
      input.value = preset[f.id];
      label.textContent = `${parseFloat(preset[f.id]).toFixed(f.decimals)} ${f.unit}`;
    }));
  });
}

function readFeatureValues(){
  const values = {};
  SECTORS.forEach(sector => sector.fields.forEach(f => {
    values[f.id] = parseFloat(document.getElementById(`in_${f.id}`).value);
  }));
  return values;
}

/* ------------------------------ Nilometer gauge --------------------------- */

function buildTicks(){
  const g = document.getElementById("tickMarks");
  for (let i = 0; i <= 10; i++){
    const y = 410 - i * 40;
    const tick = document.createElementNS("http://www.w3.org/2000/svg","line");
    tick.setAttribute("x1", 95); tick.setAttribute("x2", 102);
    tick.setAttribute("y1", y);  tick.setAttribute("y2", y);
    tick.setAttribute("class","tick");
    g.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg","text");
    label.setAttribute("x", 106); label.setAttribute("y", y+3);
    label.setAttribute("class","tick-label");
    label.textContent = (i/10).toFixed(1);
    g.appendChild(label);
  }
}

function setGauge(value){
  const rect = document.getElementById("waterRect");
  const fill = document.getElementById("waterFill");
  const totalH = 396;
  const h = value * totalH;
  rect.setAttribute("y", 410 - h);
  rect.setAttribute("height", h);

  let color = "#6FBCB3";
  if (value < 0.4) color = "#C97A5A";
  else if (value < 0.65) color = "#E4B96B";
  fill.style.fill = color;

  document.getElementById("gaugeValue").textContent = value.toFixed(3);
}

/* --------------------------------- Result --------------------------------- */

function classify(value){
  if (value < 0.40) return { tier:"low",  label:"أولوية تنموية عالية",  note:"تشير القيم إلى تحديات تنموية متعددة القطاعات؛ الوحدة مرشّحة لتصدّر أولويات التخطيط والتمويل." };
  if (value < 0.65) return { tier:"mid",  label:"أولوية تنموية متوسطة", note:"مستوى تنمية متوسط مع تفاوت بين القطاعات؛ يُنصح باستهداف القطاعات الأضعف ضمن الخطة." };
  return              { tier:"high", label:"أولوية تنموية منخفضة",  note:"مؤشرات متقدمة نسبيًا في معظم القطاعات؛ التركيز الأنسب هو الاستدامة والتحسين الحدّي." };
}

function showFactors(values){
  const stats = EDIP_DATA.stats;
  const list = document.getElementById("factorsList");
  list.innerHTML = "<div style='font-weight:700;margin-bottom:4px;'>أبرز العوامل مقارنةً بالمتوسط الوطني</div>";
  FEATURE_IMPORTANCE_ORDER.slice(0,5).forEach(fid => {
    const mean = stats[fid].mean;
    const val = values[fid];
    const diffPct = ((val - mean) / mean) * 100;
    const arrow = diffPct >= 0 ? "▲" : "▼";
    const row = document.createElement("div");
    row.className = "factor-row";
    row.innerHTML = `<span>${FEATURE_LABELS[fid]}</span><span>${arrow} ${Math.abs(diffPct).toFixed(0)}% عن المتوسط</span>`;
    list.appendChild(row);
  });
}

/* --------------------------------- Wire up --------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
  buildTicks();
  buildForm();
  buildGovSelect();

  document.getElementById("predictForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const values = readFeatureValues();
    const prediction = predictEDPI(values);
    const info = classify(prediction);

    setGauge(prediction);
    document.getElementById("gaugeLabel").textContent = info.label;

    const detail = document.getElementById("resultDetail");
    detail.hidden = false;
    const badge = document.getElementById("tierBadge");
    badge.className = "tier " + info.tier;
    badge.textContent = info.label;
    document.getElementById("resultNote").textContent = info.note;
    showFactors(values);

    detail.scrollIntoView({ behavior:"smooth", block:"nearest" });
  });
});
