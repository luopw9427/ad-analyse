import ast, json, string
SPECS = [
 ("shot_analysis","合并流程/shot_analyzer.py","PROMPT","镜头级分析","看懂每个镜头：拆四条链+卖点"),
 ("aggregate","合并流程/shot_analyzer.py","AGGREGATE_PROMPT","广告级整合","一条广告汇总成质检报告"),
 ("video_prompt","批处理/video_prompt_generator.py","PROMPT","视频级脚本","生成可投喂视频模型的完整脚本"),
 ("shot_prompt","批处理/shot_prompt_generator.py","PROMPT","单广告镜头prompt","拆出≥N条独立单镜头prompt"),
 ("common_shot","批处理/common_shot_prompt_generator.py","PROMPT","共性镜头prompt","本批次共性≥N条通用镜头prompt"),
 ("compare","批处理/batch_analyze.py","COMPARE_PROMPT","横向对比","跨广告11维度对比报告"),
 ("split_round1","拆分流程/shot_analyzer.py","ROUND1_PROMPT","拆分·第1轮Fine","拆分流程：粗分动作"),
 ("split_round2","拆分流程/shot_analyzer.py","ROUND2_PROMPT","拆分·第2轮Atomic","拆分流程：原子动作"),
 ("split_round3","拆分流程/shot_analyzer.py","ROUND3_PROMPT","拆分·第3轮卖点","拆分流程：卖点+证据"),
 ("split_aggregate","拆分流程/shot_analyzer.py","AGGREGATE_PROMPT","拆分·广告整合","拆分流程专用整合"),
]
def get_const(path, name):
    tree = ast.parse(open(path,encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id==name and isinstance(node.value, ast.Constant) and isinstance(node.value.value,str):
                    return node.value.value
    raise RuntimeError(f"没找到 {name} in {path}")
out={}
for key,path,name,label,desc in SPECS:
    text=get_const(path,name)
    req=sorted({fn for _,fn,_,_ in string.Formatter().parse(text) if fn})
    out[key]={"label":label,"desc":desc,"required":req,"content":text}
json.dump(out, open("prompts_defaults.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("生成 prompts_defaults.json，共", len(out), "个环节")
for k,v in out.items(): print(f"  {k:16s} 必需占位符: {v['required']}")
