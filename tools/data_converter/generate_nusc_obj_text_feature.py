import torch
import clip
import pickle

def build_clip():
    cond_model, preprocess = clip.load("ViT-B/32", device='cuda')
    cond_model.eval()
    return cond_model

def save_clip_feature(class_names=['unkonwn', 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian'], save_path=None):
    class_feature = {}
    cond_model = build_clip()
    with torch.no_grad():
        for i, name in enumerate(class_names):
            text_obj = clip.tokenize(name).to('cuda')
            feats_ins = cond_model.encode_text(text_obj).detach().cpu().numpy()
            class_feature.update({name: feats_ins})
    
    with open(save_path, 'wb') as f:
        pickle.dump(class_feature, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate CLIP features for object classes")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the CLIP features")
    args = parser.parse_args()

    save_clip_feature(save_path=args.save_path)