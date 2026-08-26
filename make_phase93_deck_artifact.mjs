import fs from "node:fs/promises";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const OUT = "D:/Phase93_work/docs/Phase93_答辩汇报.pptx";
const DESKTOP = "C:/Users/asus/Desktop/Phase93_答辩汇报.pptx";
const QA = "D:/Phase93_work/ppt_render_v39_rebuild";
const DIST = "C:/Users/asus/AppData/Local/Temp/codex-clipboard-b662a746-ef5f-47d3-a16e-5b7954e53a54.png";

const C = {
  bg: "#FFFFFF", ink: "#111820", muted: "#667085", line: "#CDD5DF",
  panel: "#F2F4F7", pale: "#EAF2FB", blue: "#1976E9", cyan: "#55C3F1",
  green: "#16895F", orange: "#DB9410", red: "#C83A3A", white: "#FFFFFF",
};

function shape(slide, geometry, x, y, w, h, fill = "none", lineFill = "none", lineWidth = 0, radius = undefined) {
  return slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: { style: "solid", fill: lineFill, width: lineWidth },
    ...(radius ? { borderRadius: radius } : {}),
  });
}

function textBox(slide, text, x, y, w, h, size = 20, color = C.ink, bold = false, align = "left", valign = "top") {
  const s = shape(slide, "textbox", x, y, w, h);
  s.text = text;
  s.text.style = {
    fontSize: size, color, bold, alignment: align, verticalAlignment: valign,
    autoFit: "shrinkText", typeface: "Microsoft YaHei", insets: { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return s;
}

function richText(slide, paragraphs, x, y, w, h, size = 18, color = C.ink) {
  const s = shape(slide, "textbox", x, y, w, h);
  s.text.set(paragraphs);
  s.text.style = {
    fontSize: size, color, alignment: "left", verticalAlignment: "top",
    autoFit: "shrinkText", typeface: "Microsoft YaHei", insets: { left: 0, right: 0, top: 0, bottom: 0 },
  };
  return s;
}

function header(slide, eyebrow, title, page) {
  slide.background.fill = C.bg;
  textBox(slide, eyebrow, 64, 34, 420, 22, 14, C.blue, true);
  textBox(slide, title, 64, 78, 1152, 58, 39, C.ink, true);
  shape(slide, "rect", 54, 144, 1172, 1.2, C.line);
  textBox(slide, String(page).padStart(2, "0"), 1182, 672, 34, 18, 12, C.muted, false, "right");
}

function card(slide, x, y, w, h, title, body = "", accent = C.blue, fill = C.panel, big = null) {
  shape(slide, "roundRect", x, y, w, h, fill, "none", 0, "rounded-xl");
  shape(slide, "rect", x, y, 7, h, accent);
  if (big) {
    textBox(slide, big, x + 28, y + 24, w - 48, 52, 38, accent, true);
    textBox(slide, title, x + 28, y + 82, w - 48, 34, 20, C.ink, true);
    textBox(slide, body, x + 28, y + 122, w - 48, h - 138, 15, C.muted);
  } else {
    textBox(slide, title, x + 28, y + 22, w - 48, 34, 21, C.ink, true);
    textBox(slide, body, x + 28, y + 67, w - 48, h - 82, 17, C.muted);
  }
}

function arrow(slide, x, y, w = 54, h = 34, color = C.blue) {
  shape(slide, "rightArrow", x, y, w, h, color);
}

function leftArrow(slide, x, y, w = 54, h = 34, color = C.blue) {
  shape(slide, "leftArrow", x, y, w, h, color);
}

function pill(slide, x, y, w, text, fill, color = C.white) {
  shape(slide, "roundRect", x, y, w, 34, fill, "none", 0, "rounded-full");
  textBox(slide, text, x, y + 2, w, 28, 15, color, true, "center", "middle");
}

function addTakeaway(slide, label, body, y = 588, color = C.blue) {
  shape(slide, "roundRect", 54, y, 1172, 72, C.pale, "none", 0, "rounded-xl");
  textBox(slide, label, 86, y + 19, 255, 30, 18, color, true);
  textBox(slide, body, 350, y + 17, 835, 36, 18, C.ink, true, "left", "middle");
}

function addStepRow(slide, items, y = 278) {
  const left = 86, right = 1190, gap = (right - left) / (items.length - 1);
  shape(slide, "rect", left, y + 34, right - left, 2, C.ink);
  items.forEach((it, i) => {
    const x = left + gap * i;
    shape(slide, "ellipse", x - 12, y + 22, 24, 24, it.color || C.blue);
    textBox(slide, String(i + 1).padStart(2, "0"), x - 18, y - 38, 40, 28, 16, it.color || C.blue, true, "center");
    textBox(slide, it.title, x - 66, y + 67, 150, 36, 20, C.ink, true, "center");
    textBox(slide, it.body, x - 82, y + 110, 180, 80, 15, C.muted, false, "center");
  });
}

async function writeBlob(path, blob) {
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(QA, { recursive: true });
  const p = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. Cover
  {
    const s = p.slides.add(); s.background.fill = C.bg;
    textBox(s, "Physical AI", 68, 74, 360, 32, 16, C.blue, true);
    textBox(s, "无线数字孪生信道生成", 68, 164, 880, 74, 50, C.ink, true);
    textBox(s, "Round2 · Phase93 最终端到端版本", 68, 254, 650, 38, 23, C.muted);
    shape(s, "roundRect", 68, 360, 1144, 170, C.ink, "none", 0, "rounded-xl");
    textBox(s, "核心命题", 100, 392, 180, 30, 17, C.cyan, true);
    textBox(s, "继承 Round1 的“谱预测 → 复信道重建”主线，\n针对 Round2 矩形测试岛改造成分岛验证、锚点校准与选择性修正。", 100, 438, 1010, 72, 27, C.white, true);
    textBox(s, "项目解释与答辩汇报 · 2026", 68, 652, 420, 24, 14, C.muted);
  }

  // 2. Executive summary
  {
    const s = p.slides.add(); header(s, "ROUND2 / PHASE93", "先给结论：这不是 v39 的复制，而是同一主线下的场景化升级", 2);
    card(s, 54, 184, 350, 218, "继承的主干", "PAS/PDP 评分对齐表征\n局部空间预测\n角域/时延域交替投影", C.blue, C.pale, "1 条");
    card(s, 430, 184, 350, 218, "Round2 新难点", "测试点集中在多个矩形岛\n262 个全零异常点\n随机验证严重乐观", C.orange, C.panel, "3 个");
    card(s, 806, 184, 420, 218, "Phase93 的最终策略", "11 个几何组分别建模\nP9/P10/Phase40 冻结方向\n仅修正 g5/g6 的 89 个点", C.green, C.pale, "89 / 500");
    addTakeaway(s, "一句话结论", "Round1 解决“怎样从谱恢复复信道”；Round2 进一步解决“官方矩形岛里，哪些点、用哪套局部规律、修正多少”。");
  }

  // 3. Round1 vs Round2
  {
    const s = p.slides.add(); header(s, "ROUND1 → ROUND2", "相似的是物理谱框架，真正变化的是验证几何与局部决策", 3);
    textBox(s, "Round1 v39 / v50", 80, 184, 500, 38, 24, C.blue, true);
    textBox(s, "Round2 Phase93", 700, 184, 500, 38, 24, C.green, true);
    const rows = [
      ["验证", "固定五折，每折 200 点", "5 折矩形挖空 + 官方人口加权"],
      ["空间模型", "15 个克里金专家 + 分组门控", "11 个几何组 + PAS/PDP 独立邻域"],
      ["地图", "209 维走廊/环形/天际线描述", "LOS/local/rich 特征进入岛内距离与校准"],
      ["复信道", "物理相位初值 + 交替投影", "保留交替投影，幅度采用保守规范"],
      ["最终校准", "none/PAS/PDP/PAS+PDP 四动作", "锚点残差 + anti-P10 + g5/g6 选择性修正"],
    ];
    rows.forEach((r, i) => {
      const y = 236 + i * 66;
      shape(s, "roundRect", 54, y, 1172, 54, i % 2 ? C.bg : C.panel, C.line, 1, "rounded-md");
      textBox(s, r[0], 76, y + 13, 140, 28, 17, C.ink, true);
      textBox(s, r[1], 238, y + 10, 420, 34, 17, C.muted);
      textBox(s, r[2], 684, y + 10, 514, 34, 17, C.ink, true);
    });
    addTakeaway(s, "答辩重点", "不要把 Phase93 讲成 57 个脚本；要讲成“同一谱重建底座上，Round2 新增四层场景适配”。", 592, C.orange);
  }

  // 4. Metrics
  {
    const s = p.slides.add(); header(s, "任务与评价", "PAS、PDP 与 NMSE 分别约束角度、时延和完整复数误差", 4);
    card(s, 54, 180, 360, 290, "PAS · 角度功率谱", "256 端口 → 2×16×8\n沿 H/V 阵列做 FFT2\n比较归一化角域功率形状", C.blue);
    card(s, 460, 180, 360, 290, "PDP · 时延功率谱", "192 子载波处于频率域\n沿频率维做 FFT\n比较归一化时延功率形状", C.green);
    card(s, 866, 180, 360, 290, "NMSE · 原始复信道", "Σ|Ĥ−H|² / Σ|H|²\n同时敏感于幅度和相位\n越接近 0 越好", C.red);
    pill(s, 104, 492, 260, "40% PAS", C.blue); pill(s, 510, 492, 260, "40% PDP", C.green); pill(s, 916, 492, 260, "20% · 1/(1+NMSE)", C.red);
    addTakeaway(s, "计算例子", "PAS=0.80、PDP=0.70、NMSE=1.00 → 总分 = 0.32 + 0.28 + 0.10 = 0.70。", 584);
  }

  // 5. Data
  {
    const s = p.slides.add(); header(s, "数据分析", "4000 个训练点并不等于 4000 个有效参考点，测试集也不是随机采样", 5);
    try {
      const b = await fs.readFile(DIST);
      s.images.add({ blob: b, contentType: "image/png", alt: "Round2 train and clustered test point distribution", fit: "contain", position: { left: 54, top: 176, width: 610, height: 370 }, geometry: "roundRect", borderRadius: "rounded-xl" });
    } catch {
      shape(s, "roundRect", 54, 176, 610, 370, C.panel, C.line, 1, "rounded-xl");
      textBox(s, "训练 / 测试点空间分布图", 140, 330, 440, 40, 24, C.muted, true, "center");
    }
    card(s, 700, 176, 244, 162, "训练位置", "Round2_Train_Pos.npy", C.blue, C.pale, "4000");
    card(s, 970, 176, 244, 162, "测试位置", "多个小矩形集中分布", C.orange, C.panel, "500");
    card(s, 700, 364, 244, 162, "全零异常", "energy=0，统一剔除", C.red, C.panel, "262");
    card(s, 970, 364, 244, 162, "有效训练", "进入邻域、锚点和验证", C.green, C.pale, "3738");
    addTakeaway(s, "数据决定模型", "若随机划分验证点，周围仍有密集训练邻居，会系统性高估官方矩形空洞上的泛化能力。", 574, C.orange);
  }

  // 6. Validation geometry
  {
    const s = p.slides.add(); header(s, "Round2 改变 1 / 验证", "验证集必须模拟官方矩形空洞，而不是从 4000 点中随机抽样", 6);
    shape(s, "roundRect", 54, 182, 600, 352, C.panel, C.line, 1, "rounded-xl");
    textBox(s, "训练点平面（示意）", 84, 204, 260, 26, 17, C.muted, true);
    for (let i = 0; i < 95; i++) {
      const x = 90 + ((i * 83) % 520), y = 250 + ((i * 47) % 240);
      shape(s, "ellipse", x, y, 5, 5, C.blue);
    }
    [[148,278,92,70],[318,250,110,86],[454,362,125,78],[242,402,92,65]].forEach((r, i) => {
      shape(s, "rect", ...r, i % 2 ? "#FFF2D6" : "#FCE1E1", C.red, 2);
      textBox(s, "VAL", r[0] + 22, r[1] + r[3] / 2 - 10, r[2] - 44, 20, 13, C.red, true, "center");
    });
    card(s, 690, 182, 536, 108, "① 恢复官方几何", "DBSCAN 恢复分组标签和包围盒", C.blue, C.pale);
    card(s, 690, 304, 536, 108, "② 在训练点中挖矩形洞", "匹配宽高/人口，并增加 3 m buffer", C.orange, C.panel);
    card(s, 690, 426, 536, 108, "③ 固定 5 折 + 官方权重", "固定种子；各岛按官方测试人口加权", C.green, C.pale);
    addTakeaway(s, "验证原则", "训练过程可以复杂，但选型依据必须来自无泄漏的矩形 OOF，而不是随机点分数。", 576);
  }

  // 7. End-to-end
  {
    const s = p.slides.add(); header(s, "端到端流程", "从四个赛事原始文件出发，把数据、谱预测、通道重建和冻结校准串成闭环", 7);
    addStepRow(s, [
      {title:"原始输入", body:"Train H / Train Pos\nTest Pos / Map.ply"},
      {title:"清洗与特征", body:"剔除 262 零信道\nLOS / local / rich"},
      {title:"空间谱预测", body:"分组 PAS/PDP 邻域\nPhase6 / P9 / P10"},
      {title:"复信道重建", body:"角域/时延域\n交替幅度投影"},
      {title:"冻结修正", body:"anti-P10\ng5/g6 full-192", color:C.green},
      {title:"输出校验", body:"complex64 / finite\n500×256×4×192", color:C.green},
    ], 242);
    addTakeaway(s, "推理入口", "python build_phase93_end_to_end.py → Round2_Test_Channel_phase93_g56_antip10_plus_symmetric_clamp.npy", 590);
  }

  // 8. Inherited backbone
  {
    const s = p.slides.add(); header(s, "继承自 v39 的主干", "先把超高维复信道变成可解释谱，再用物理结构恢复完整 H", 8);
    card(s, 54, 188, 250, 294, "复信道 H", "256 BS × 4 UE × 192\n每点 196,608 个复数", C.red, C.panel);
    arrow(s, 324, 316);
    card(s, 398, 188, 250, 132, "PAS 分支", "阵列 FFT2\n角度功率形状", C.blue, C.pale);
    card(s, 398, 350, 250, 132, "PDP 分支", "子载波 FFT\n时延功率形状", C.green, C.pale);
    arrow(s, 670, 316);
    card(s, 742, 188, 216, 294, "空间预测", "位置邻域\n地图上下文\n分组参数\n锚点残差", C.orange, C.panel);
    arrow(s, 980, 316);
    card(s, 1048, 188, 178, 294, "重建 Ĥ", "保留相位\n替换幅度\n交替投影", C.green, C.pale);
    addTakeaway(s, "共同哲学", "Round1 与 Round2 都没有直接回归 19.7 万维复数；它们都把评分结构前置到模型表征中。", 578);
  }

  // 9. Island specialization
  {
    const s = p.slides.add(); header(s, "Round2 改变 2 / 分组建模", "11 个几何组不是一个统一空间场：PAS 与 PDP 甚至需要不同邻域", 9);
    shape(s, "roundRect", 54, 184, 470, 360, C.panel, C.line, 1, "rounded-xl");
    textBox(s, "几何组标签（示意）", 82, 206, 250, 28, 18, C.muted, true);
    const islands = [[92,264,100,64,0],[218,246,74,54,1],[344,272,92,70,3],[116,382,118,82,4],[270,382,74,72,5],[378,392,104,78,6]];
    islands.forEach(([x,y,w,h,g],i)=>{shape(s,"roundRect",x,y,w,h,i%2?"#FFF2D6":C.pale,i%2?C.orange:C.blue,2,"rounded-md");textBox(s,`g${g}`,x,y+18,w,26,17,i%2?C.orange:C.blue,true,"center");});
    card(s, 566, 184, 300, 170, "PAS 邻域", "每岛独立选择 k / 距离幂 / 广义均值 q\n可启用角谱对齐、仿射与二次矩约束", C.blue, C.pale);
    card(s, 896, 184, 330, 170, "PDP 邻域", "可使用完全不同的 k / 距离幂 / 地图度量\n避免角域最优邻居强迫时延域服从", C.green, C.pale);
    card(s, 566, 382, 660, 162, "为什么能提高上限", "矩形内部、矩形边缘、遮挡边界和不同基站侧的空间平滑尺度不同。分组参数把一个全局偏差问题拆成 11 个可验证的局部问题。", C.orange, C.panel);
    addTakeaway(s, "核心变化", "Round1 是多专家选择；Round2 进一步把“专家适用域”显式绑定到官方测试几何。", 578);
  }

  // 10. Map features
  {
    const s = p.slides.add(); header(s, "Round2 改变 3 / 地图上下文", "点云不做完整射线追踪，而是压缩成影响邻域选择的传播环境特征", 10);
    card(s, 54, 184, 270, 310, "LOS 特征", "基站到 UE 路径上的\n遮挡高度、密度、墙面\n左右侧分别编码", C.blue, C.pale);
    card(s, 350, 184, 270, 310, "Local 特征", "测试点周围局部高度\n墙面/地面密度\n多尺度邻域统计", C.orange, C.panel);
    card(s, 646, 184, 270, 310, "Rich 特征", "方向、角度、障碍与\n频带描述联合进入\n树模型/门控模型", C.green, C.pale);
    card(s, 942, 184, 284, 310, "如何使用", "不是直接生成 H\n而是改变谁是邻居、\n残差应修多少、\n哪个岛采用哪套参数", C.red, C.panel);
    addTakeaway(s, "与 v39 的关系", "v39 的 209 维地图描述更系统；Round2 保留“地图决定传播相似度”的思想，并围绕岛内验证做更强的任务定制。", 576);
  }

  // 11. Base reconstruction
  {
    const s = p.slides.add(); header(s, "基础模型", "局部邻居给出目标谱，交替投影把 PAS 与 PDP 约束写回复信道", 11);
    card(s, 54, 196, 280, 260, "局部邻居", "cKDTree 查询候选\nw ∝ 1/max(d,0.25)^p\n仿射/二次矩修正", C.blue, C.pale);
    arrow(s, 354, 306);
    card(s, 430, 196, 280, 260, "目标谱", "PAS：角域功率\nPDP：时延域功率\n分别聚合、分别归一化", C.orange, C.panel);
    arrow(s, 730, 306);
    shape(s, "roundRect", 806, 196, 420, 260, C.pale, "none", 0, "rounded-xl");
    textBox(s, "交替投影闭环", 840, 220, 350, 34, 22, C.ink, true, "center");
    pill(s, 854, 282, 118, "BS-FFT", C.blue); pill(s, 1058, 282, 118, "Freq-FFT", C.green);
    arrow(s, 982, 282, 58, 34, C.orange); leftArrow(s, 982, 350, 58, 34, C.orange);
    textBox(s, "替换幅度，保留当前复相位", 858, 382, 330, 34, 17, C.muted, true, "center");
    addTakeaway(s, "物理含义", "PAS 和 PDP 是两个不完全兼容的边缘约束；有限轮交替投影在两者之间取得可控折中。", 576);
  }

  // 12. Phase6
  {
    const s = p.slides.add(); header(s, "Phase6 / 物理谱底座", "Phase6 把位置、方向和地图描述变成 24-band PAS 目标", 12);
    const xs=[54,264,474,684];
    [["位置与岛标签","x / y / side / group",C.blue],["方向特征","水平/垂直 canonical",C.orange],["地图特征","LOS / local / rich",C.green],["信道描述","24-band PAS/PDP",C.red]].forEach((r,i)=>card(s,xs[i],196,184,190,r[0],r[1],r[2],i%2?C.panel:C.pale));
    arrow(s, 878, 274, 62, 38, C.blue);
    card(s, 960, 196, 266, 190, "冻结组合", "MLP / ExtraTrees / canonical\nrich-tree / gate\n按矩形 OOF 选型", C.blue, C.panel);
    shape(s, "roundRect", 164, 438, 952, 108, C.ink, "none", 0, "rounded-xl");
    textBox(s, "Phase6 决定最终结果的大部分空间结构；P9 与 Phase10 都从它出发，只做可验证的局部方向修正。", 206, 470, 870, 44, 22, C.white, true, "center");
    addTakeaway(s, "叙述方式", "把 Phase6 讲成“共同底座”，后续版本讲成“在底座上增加哪种可解释修正”，主线会比按脚本编号清楚。", 590);
  }

  // 13. P9
  {
    const s = p.slides.add(); header(s, "P9 / 锚点联合模型", "矩形内部的训练点成为官方几何锚点，负责校正局部 PAS 与 PDP 偏差", 13);
    card(s, 54, 184, 250, 320, "外部训练点", "先排除矩形内锚点\n构造无泄漏局部基线", C.blue, C.pale);
    arrow(s, 320, 322);
    card(s, 390, 184, 250, 320, "锚点残差", "PAS：log 真值/基线\nPDP：log 真值/基线\n按岛内邻居插值", C.orange, C.panel);
    arrow(s, 656, 322);
    card(s, 726, 184, 250, 320, "收益门控", "ExtraTrees 学习 alpha\n五折收益网格决定\n是否靠近局部目标", C.green, C.pale);
    arrow(s, 992, 322);
    card(s, 1062, 184, 164, 320, "P9", "12 次\nPAS/PDP\n联合投影", C.red, C.panel);
    addTakeaway(s, "关键防泄漏", "锚点先用矩形外训练点预测，再计算残差；锚点真值只用于构造可推理的岛内校准场。", 576);
  }

  // 14. P10/Phase40
  {
    const s = p.slides.add(); header(s, "Phase10 → Phase40", "P10 不是最终答案，而是构造一条“已知较差方向”，再用官方反馈反向移动", 14);
    shape(s, "roundRect", 54, 184, 1172, 166, C.panel, "none", 0, "rounded-xl");
    pill(s, 86, 240, 160, "P9 · 0.6395", C.blue); arrow(s, 274, 240, 102, 36, C.orange); pill(s, 408, 240, 190, "P10 · 0.6354", C.red);
    textBox(s, "d = clip(log(PAS₁₀ / PAS₉), −2, 2)", 642, 232, 520, 52, 23, C.ink, true, "center");
    shape(s, "roundRect", 54, 386, 1172, 166, C.pale, "none", 0, "rounded-xl");
    pill(s, 86, 442, 160, "P9", C.blue); arrow(s, 274, 442, 102, 36, C.green); pill(s, 408, 442, 238, "Phase40 · anti-P10", C.green);
    textBox(s, "PAS₄₀ = normalize(PAS₉ · exp(−0.50 d))", 690, 434, 468, 52, 23, C.ink, true, "center");
    addTakeaway(s, "为什么成立", "官方分数表明 P10 方向持续变差；Phase40 将这条黑盒反馈冻结成可复现的反向谱修正，而不再追逐更高本地分数。", 582, C.orange);
  }

  // 15. Phase93
  {
    const s = p.slides.add(); header(s, "Phase93 / 最大上限的选择性修正", "只对 g5/g6 的 89 个点使用 full-192 锚点目标，其余 411 点锁定 Phase40", 15);
    card(s, 54, 184, 360, 224, "g5 + g6", "43 + 46 = 89 个测试点\n22 + 42 个训练锚点\n4 邻居，距离平方权重", C.blue, C.pale, "89");
    arrow(s, 438, 276, 62, 38, C.orange);
    card(s, 526, 184, 360, 224, "full-192 PAS", "目标 = 0.80×P9 + 0.20×Anchor\n12 次角域投影\n随后仍应用 anti-P10", C.orange, C.panel, "192");
    arrow(s, 910, 276, 62, 38, C.green);
    card(s, 998, 184, 228, 224, "其余点", "411 行逐位等于\nPhase40\n限制修正外溢", C.green, C.pale, "411");
    shape(s, "roundRect", 54, 444, 1172, 102, C.ink, "none", 0, "rounded-xl");
    textBox(s, "C93[i,:,u,:] = s90(P9[i],u) × C89[i,:,u,:]；正式测试中所有 s90=1，因此最终数值就是 g5/g6 + anti-P10 结果。", 88, 473, 1104, 48, 20, C.white, true, "center");
    addTakeaway(s, "设计思想", "当提交次数有限时，优先扩大有矩形 OOF 证据的局部修正，不把风险扩散到全部 500 点。", 586);
  }

  // 16. Clamp audit
  {
    const s = p.slides.add(); header(s, "Phase93 事实审计", "clamp-floor 是安全兜底；正式 500 点输出中没有改变任何数值", 16);
    card(s, 54, 190, 340, 260, "计算规则", "从原始 P9 预测计算\n每样本×UE 的 PAS/PDP\n1% 低尾范数 q", C.blue, C.pale);
    arrow(s, 416, 302);
    card(s, 500, 190, 340, 260, "尺度规则", "s = clip(√(1e−15/q), 1, 5)\n只允许抬升极低能量分支\n不读取测试真值", C.orange, C.panel);
    arrow(s, 862, 302);
    card(s, 946, 190, 280, 260, "正式结果", "2000 / 2000 分支 s=1\nactive branches = 0\nPhase93 = Phase89", C.green, C.pale, "0");
    addTakeaway(s, "答辩口径", "clamp-floor 不是 Phase93 得分提升来源；真正的最终改动只有 g5/g6 full-192 锚点修正与冻结 anti-P10。", 576, C.red);
  }

  // 17. 57 stages
  {
    const s = p.slides.add(); header(s, "训练与复现", "57 个执行阶段可以归纳为 6 个职责清晰的宏模块", 17);
    const stages=[
      ["01", "数据/地图", "LOS · local · rich\n信道描述缓存", C.blue],
      ["02", "矩形 OOF", "5 折挖空\n官方人口加权", C.orange],
      ["03", "基础谱搜索", "Phase1–6\n分组参数与目标", C.blue],
      ["04", "锚点与组合", "Phase7–10\n残差/门控/组合", C.green],
      ["05", "官方反馈", "Phase40\n冻结 anti-P10", C.red],
      ["06", "选择性修正", "Phase93\ng5/g6 full-192", C.green],
    ];
    stages.forEach((r,i)=>{const x=54+i*195;card(s,x,194,170,314,r[1],r[2],r[3],i%2?C.panel:C.pale);if(i<5)arrow(s,x+174,320,20,26,C.line);});
    addTakeaway(s, "工程结论", "脚本多是为了保存 OOF 证据、冻结参数和可恢复缓存；算法主线仍是 6 个宏模块，而不是 57 个彼此独立的模型。", 578);
  }

  // 18. Results
  {
    const s = p.slides.add(); header(s, "验证结果与可信度", "Phase93 的证据是局部矩形 OOF 增益与严格字节级约束，不把本地分数冒充官方分数", 18);
    card(s, 54, 184, 270, 232, "相对 Phase40", "五折矩形 OOF 的平均增益", C.green, C.pale, "+0.000806");
    card(s, 350, 184, 270, 232, "正增益折数", "5 折中 4 折提高\n最差折 −0.000130", C.blue, C.panel, "4 / 5");
    card(s, 646, 184, 270, 232, "稳健下界", "mean − 0.75×std", C.orange, C.pale, "+0.000272");
    card(s, 942, 184, 284, 232, "已知官方参照", "Phase40 官方测试\n注意：不是 Phase93 官方分", C.red, C.panel, "0.639697");
    shape(s, "roundRect", 54, 454, 1172, 94, C.ink, "none", 0, "rounded-xl");
    textBox(s, "输出 QA：shape=(500,256,4,192) · complex64 · 786,432,128 bytes · finite · 500 行全部非零 · SHA256 固定", 82, 485, 1116, 36, 20, C.white, true, "center");
    addTakeaway(s, "结果口径", "仓库未记录 Phase93 的官方隐藏测试分数，因此汇报只陈述可复现的矩形 OOF 增益和已知 Phase40 官方参照。", 586);
  }

  // 19. Reproduction
  {
    const s = p.slides.add(); header(s, "端到端复现与提交", "赛事方只需补齐四个官方原始文件，即可从零生成 Phase93 输出", 19);
    card(s, 54, 184, 360, 310, "需要的赛事原始数据", "Round2_Train_Channel.npy\nRound2_Train_Pos.npy\nRound2_Test_Pos.npy\nRound2_Map.ply", C.blue, C.pale);
    arrow(s, 438, 318, 70, 40, C.orange);
    card(s, 532, 184, 360, 310, "运行命令", "python -m pip install -r requirements.txt\n\npython build_phase93_end_to_end.py\n\n可用 --list-stages / --from-stage 恢复", C.orange, C.panel);
    arrow(s, 916, 318, 70, 40, C.green);
    card(s, 1010, 184, 216, 310, "Phase93 NPY", "Round2_Test_Channel_...\n\n500×256×4×192\ncomplex64", C.green, C.pale);
    addTakeaway(s, "提交边界", "提交源码、requirements、冻结 manifest 与最终 NPY；赛事原始训练信道、位置和点云不重复上传。", 576);
  }

  // 20. Summary
  {
    const s = p.slides.add(); header(s, "总结", "Phase93 的价值不在堆版本，而在把 Round1 主线变成适配 Round2 官方几何的可复现系统", 20);
    card(s, 54, 190, 350, 250, "继承", "PAS/PDP 评分对齐\n局部空间连续性\n复信道交替投影", C.blue, C.pale);
    card(s, 430, 190, 350, 250, "改变", "矩形挖空 OOF\n11 组独立参数\n锚点残差与官方反馈方向", C.orange, C.panel);
    card(s, 806, 190, 420, 250, "最终选择", "只在 g5/g6 扩大有证据的修正\n89 点改变、411 点锁定\n四原始文件可端到端复现", C.green, C.pale);
    textBox(s, "一句话总结", 64, 492, 220, 30, 17, C.blue, true);
    textBox(s, "先理解数据与指标，再按官方空间几何拆解模型，比盲目扩大黑盒模型更有效。", 64, 538, 1100, 52, 30, C.ink, true);
  }

  for (const [i, s] of p.slides.items.entries()) {
    const stem = `slide-${String(i + 1).padStart(2, "0")}`;
    await writeBlob(`${QA}/${stem}.png`, await p.export({ slide: s, format: "png", scale: 1 }));
    await fs.writeFile(`${QA}/${stem}.layout.json`, await (await s.export({ format: "layout" })).text());
  }
  await writeBlob(`${QA}/deck-montage.webp`, await p.export({ format: "webp", montage: true, scale: 0.6 }));
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(OUT);
  await fs.copyFile(OUT, DESKTOP);
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
