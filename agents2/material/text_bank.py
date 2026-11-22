"""Текстовые эмбеддинги общих словарей материалов для CLIP-L (dfn2b) и SigLIP-B/16.
Словари — общие (названия материалов, типичный мусор из каждого материала, свойства поверхности, упаковка с этикетками),
не подогнаны под конкретные предметы. Для каждого набора сохраняется усреднённый по шаблонам эмбеддинг класса
и отдельные эмбеддинги каждой формулировки. Модели грузятся по одной (GPU < 2 ГБ). Кэш: text_bank.npz"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

HERE = Path(__file__).resolve().parent
MATS = ["metal", "wood", "plastic", "glass", "paper", "mineral"]
MODELS = {"clipL": ("ViT-L-14", "dfn2b"), "siglip": ("ViT-B-16-SigLIP", "webli")}
DEV = "cuda" if torch.cuda.is_available() else "cpu"

TEMPLATES = {
    "floor": ["a photo of {} on the floor.", "a close-up photo of {}.", "{} lying on a tiled floor.", "trash made of {}.", "a photo of an object made of {}."],
    "plain": ["a photo of {}.", "{}.", "a picture of {}.", "a photo of {} on the ground."],
    "robot": ["a top-down photo of {} on an indoor floor.", "litter on the floor: {}.", "a piece of garbage, {}, seen from a cleaning robot.", "{} dropped on the floor."],
    "light": ["a dim photo of {} on the floor.", "a brightly lit photo of {}.", "a photo of {} under side lighting with shadows.", "a low-light photo of {} on a tiled floor."],
}

SETS = {
    # прежние наборы из full_pipeline (для сравнения)
    "generic_syn": {"metal": ["metal", "metallic object"], "wood": ["wood", "wooden object"], "plastic": ["plastic", "plastic object"],
                    "glass": ["glass", "glass object"], "paper": ["paper", "paper or cardboard object"], "mineral": ["stone", "mineral or stone object"]},
    "rich": {"metal": ["metal", "metallic object", "a metal can", "an aluminium can", "a dark painted metal tray", "a steel plate", "a metal bar", "metal screws and bolts", "an aluminium profile"],
             "wood": ["wood", "wooden object", "a wooden plank", "a wooden stick", "wood chips", "a piece of timber", "a wooden block"],
             "plastic": ["plastic", "plastic object", "a plastic bottle", "a plastic bag", "a plastic container", "plastic packaging", "plastic film"],
             "glass": ["glass", "glass object", "a glass bottle", "a brown glass bottle", "a dark amber glass bottle with a label", "a clear glass bottle", "a glass jar"],
             "paper": ["paper", "paper or cardboard object", "a cardboard box", "a flat cardboard sheet", "a paper bag", "a paper cup", "crumpled paper"],
             "mineral": ["stone", "mineral or stone object", "a rock", "a stone slab", "a brick", "gravel", "a concrete block"]},
    # общие словари
    "names": {"metal": ["metal", "a metal object", "something made of metal", "metallic"],
              "wood": ["wood", "a wooden object", "something made of wood", "wooden"],
              "plastic": ["plastic", "a plastic object", "something made of plastic", "synthetic plastic"],
              "glass": ["glass", "a glass object", "something made of glass", "glassware"],
              "paper": ["paper", "a paper object", "something made of paper or cardboard", "cardboard"],
              "mineral": ["stone", "a stone object", "something made of stone, concrete or ceramic", "mineral"]},
    "waste": {"metal": ["a tin can", "an aluminium beverage can", "an aerosol spray can", "a metal lid", "a metal bottle cap", "screws, nails and bolts", "a piece of metal wire",
                        "a metal pipe", "crumpled aluminium foil", "a metal tray", "a steel bracket", "a metal tool", "a metal rod", "scrap metal"],
              "wood": ["a wooden plank", "a wooden stick", "a tree branch", "wood chips", "sawdust", "a wooden block", "a broken piece of furniture made of wood", "a twig",
                       "tree bark", "a piece of plywood", "a wooden board", "a wooden pallet fragment", "a wooden toy", "a piece of timber"],
              "plastic": ["a plastic bottle", "a plastic bag", "a plastic cup", "a plastic wrapper", "a plastic food container", "a plastic lid", "plastic film",
                          "a piece of styrofoam", "a plastic toy", "a plastic pipe", "a plastic canister", "a plastic tray", "a plastic jerrycan", "a plastic bucket"],
              "glass": ["a glass bottle", "a glass jar", "a shard of glass", "broken glass", "a drinking glass", "a wine bottle", "a beer bottle", "a glass vase",
                        "a glass cup", "a glass lid", "a glass plate", "a perfume bottle made of glass", "a glass bowl", "a glass tube"],
              "paper": ["a cardboard box", "a newspaper", "a paper bag", "crumpled paper", "a paper cup", "a carton", "a magazine", "a paper tissue", "a sheet of paper",
                        "a corrugated cardboard sheet", "a paper envelope", "a paper plate", "a cardboard tube", "a book"],
              "mineral": ["a stone", "a rock", "a brick", "a piece of concrete", "gravel", "a ceramic tile", "a pebble", "a chunk of plaster", "sand", "a piece of cement",
                          "a broken ceramic cup", "a paving stone", "crushed stone", "a clay fragment"]},
    "surface": {"metal": ["a shiny metal surface", "brushed steel", "a painted metal surface", "rusty iron", "dented aluminium", "a matte black metal surface", "galvanized steel",
                          "a metallic reflective surface", "a dark grey metal object with specular highlights", "a thin sheet metal object"],
                "wood": ["a surface with wood grain", "rough sawn timber", "light brown natural wood", "varnished wood", "a splintered wooden surface", "weathered grey wood",
                         "dark brown wood", "a knotty wooden surface", "a fibrous wooden texture", "a plain wooden surface"],
                "plastic": ["a glossy plastic surface", "a translucent plastic surface", "printed plastic packaging", "a matte colored plastic surface", "a thin crinkled plastic film",
                            "a transparent plastic surface", "a flexible plastic object", "a molded plastic object", "a brightly colored plastic object", "a white plastic surface"],
                "glass": ["a transparent glass surface", "a reflective glass surface", "amber colored glass", "green glass", "clear glass with reflections", "frosted glass",
                          "a smooth glossy glass object", "dark brown translucent glass", "a fragile glass object", "a thick glass object"],
                "paper": ["a matte fibrous paper surface", "corrugated cardboard", "brown kraft paper", "white printed paper", "a soft creased paper surface", "a torn paper edge",
                          "a thin flat paper sheet", "a folded cardboard surface", "glossy printed paper", "grey recycled cardboard"],
                "mineral": ["a rough stone surface", "grey concrete", "a porous mineral surface", "a red clay brick surface", "a glazed ceramic surface", "a dusty rock",
                            "a grainy stone texture", "a heavy solid rock", "a chipped concrete surface", "a smooth pebble surface"]},
    "packaging": {"metal": ["a metal can with a printed label", "a tin can with a paper label", "a metal spray can with a printed design", "an aluminium can with colorful print",
                            "a metal container with a label wrapped around it", "a metal tin with a lid"],
                  "wood": ["a wooden crate", "a wooden box", "a wooden container", "wooden packaging", "a wooden case", "a small wooden barrel"],
                  "plastic": ["a plastic bottle with a printed label", "a plastic container with a label", "a squeezable plastic bottle", "a plastic bottle with a screw cap",
                              "a plastic jug with a label", "shrink-wrapped plastic packaging"],
                  "glass": ["a glass bottle with a paper label", "a brown glass beer bottle with a label", "a glass jar with a label", "a dark glass bottle with a printed label",
                            "a green glass bottle with a label", "a glass bottle with a metal cap"],
                  "paper": ["a cardboard box with printed text", "a paper bag with a logo", "a cardboard package", "a paper carton with print", "a folded cardboard box", "a paper wrapper"],
                  "mineral": ["a ceramic pot", "a stone container", "a clay jar", "a ceramic bowl", "a concrete block", "a terracotta pot"]},
    "shape_free": {"metal": ["metal", "iron", "steel", "aluminium", "tin", "copper", "brass", "a metal alloy"],
                   "wood": ["wood", "timber", "lumber", "pine wood", "oak wood", "plywood", "hardwood", "softwood"],
                   "plastic": ["plastic", "polyethylene", "PET plastic", "polypropylene", "polystyrene", "PVC", "nylon plastic", "acrylic plastic"],
                   "glass": ["glass", "soda-lime glass", "bottle glass", "window glass", "tempered glass", "crystal glass", "amber glass", "borosilicate glass"],
                   "paper": ["paper", "cardboard", "paperboard", "kraft paper", "newsprint", "corrugated board", "tissue paper", "pulp paper"],
                   "mineral": ["stone", "concrete", "brick", "ceramic", "granite", "limestone", "cement", "gravel"]},
    # объединённый общий словарь: названия + типичный мусор + поверхности (без packaging), для lse
    "broad": {
        "metal": ["metal", "a metal object", "metallic", "a tin can", "an aluminium can", "an aerosol spray can", "a metal lid", "screws, nails and bolts", "a metal pipe",
                  "crumpled aluminium foil", "a metal tray", "a steel bracket", "a metal rod", "scrap metal", "a shiny metal surface", "brushed steel", "a painted metal surface",
                  "rusty iron", "a matte black metal surface", "an aluminium profile", "a steel plate", "a metal bar", "a metal can with a printed label"],
        "wood": ["wood", "a wooden object", "wooden", "a wooden plank", "a wooden stick", "a tree branch", "wood chips", "sawdust", "a wooden block", "a twig", "tree bark",
                 "a piece of plywood", "a wooden board", "a piece of timber", "a surface with wood grain", "rough sawn timber", "light brown natural wood", "varnished wood",
                 "a splintered wooden surface", "a wooden crate"],
        "plastic": ["plastic", "a plastic object", "synthetic plastic", "a plastic bottle", "a plastic bag", "a plastic cup", "a plastic wrapper", "a plastic food container",
                    "a plastic lid", "plastic film", "a piece of styrofoam", "a plastic toy", "a plastic pipe", "a plastic canister", "a plastic tray", "a plastic bucket",
                    "a glossy plastic surface", "a translucent plastic surface", "printed plastic packaging", "a matte colored plastic surface", "a black plastic part",
                    "a molded plastic object", "foam plastic", "a plastic bottle with a printed label"],
        "glass": ["glass", "a glass object", "glassware", "a glass bottle", "a glass jar", "a shard of glass", "broken glass", "a drinking glass", "a wine bottle", "a beer bottle",
                  "a brown glass bottle", "a green glass bottle", "a clear glass bottle", "a transparent glass surface", "a reflective glass surface", "amber colored glass",
                  "a glass bottle with a paper label", "a dark glass bottle with a printed label", "a smooth glossy glass object"],
        "paper": ["paper", "a paper object", "cardboard", "a cardboard box", "a newspaper", "a paper bag", "crumpled paper", "a paper cup", "a carton", "a magazine",
                  "a paper tissue", "a sheet of paper", "a corrugated cardboard sheet", "a paper envelope", "a cardboard tube", "a matte fibrous paper surface",
                  "brown kraft paper", "white printed paper", "a soft creased paper surface", "a flat cardboard sheet"],
        "mineral": ["stone", "a stone object", "mineral", "a rock", "a brick", "a piece of concrete", "gravel", "a ceramic tile", "a pebble", "a chunk of plaster", "sand",
                    "a piece of cement", "a broken ceramic cup", "a paving stone", "crushed stone", "a rough stone surface", "grey concrete", "a porous mineral surface",
                    "a red clay brick surface", "a glazed ceramic surface", "a stone slab", "a concrete block"]},
}


@torch.no_grad()
def main():
    import open_clip
    out = {}
    info = {}
    for tag, (name, pre) in MODELS.items():
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pre, device=DEV)
        model.eval()
        tok = open_clip.get_tokenizer(name)
        for sname, prompts in SETS.items():
            for tname, temps in TEMPLATES.items():
                cls_embs, per = [], []
                for mt in MATS:
                    txt = [t.format(p) for p in prompts[mt] for t in temps]
                    E = []
                    for i in range(0, len(txt), 64):
                        E.append(Fn.normalize(model.encode_text(tok(txt[i:i + 64]).to(DEV)).float(), dim=-1))
                    E = torch.cat(E).view(len(prompts[mt]), len(temps), -1)
                    per_prompt = Fn.normalize(E.mean(1), dim=-1)
                    per.append(per_prompt.cpu().numpy())
                    cls_embs.append(Fn.normalize(Fn.normalize(E.view(-1, E.shape[-1]).mean(0), dim=0), dim=0).cpu().numpy())
                out[f"{tag}|{sname}|{tname}"] = np.stack(cls_embs)
                for c, mt in enumerate(MATS):
                    out[f"{tag}|{sname}|{tname}|per|{mt}"] = per[c]
                info[f"{sname}|{tname}"] = {mt: len(prompts[mt]) for mt in MATS}
        del model
        torch.cuda.empty_cache()
        print("done", tag, flush=True)
    np.savez_compressed(HERE / "text_bank.npz", **out)
    json.dump({"sets": SETS, "templates": TEMPLATES}, open(HERE / "text_bank.json", "w"), indent=1)
    print("saved", len(out), "arrays")


if __name__ == "__main__":
    main()
