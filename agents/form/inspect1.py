import json
objs={}
meta=json.load(open("/home/cleaner/Cleaner/audit/cache/meta.json"))
for m in meta:
    if m["src"]=="real_train": objs.setdefault(m["oid"],m["form"])
for f in (1,2,3):
    p={1:"/home/cleaner/Cleaner/configs/object_id_split.json",2:"/home/cleaner/Cleaner/checkpoints/score83_cv_fold2_noval18/configs/object_id_split.json",3:"/home/cleaner/Cleaner/checkpoints/score83_cv_fold3_noval18/configs/object_id_split.json"}[f]
    s=json.load(open(p)); print(f, {k:(len(v) if hasattr(v,"__len__") else v) for k,v in s.items()}); print("  val objs", s["validation_object_ids"], [objs[o] for o in s["validation_object_ids"]])
r=json.load(open("/home/cleaner/Cleaner/audit/cache/full_pipeline_report.json"))
print(json.dumps(r["info"],indent=0)[:1500]); k=list(r["summary"])[0]; print(k, r["summary"][k])
