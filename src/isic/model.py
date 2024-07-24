import torch
from open_clip import CustomTextCLIP
class ClipModel(torch.nn.Module):
    def __init__(self, clip_model: CustomTextCLIP):
        super(ClipModel, self).__init__()
        self.module = clip_model
        self.output_dict = True
        self.module.output_dict = True

    def forward(self, image, text, secondary_text=None) -> dict:
        image_features = (
            self.module.encode_image(image, normalize=True)
            if image is not None
            else None
        )
        text_features = (
            self.module.encode_text(text, normalize=True) if text is not None else None
        )
        secondary_text_features = None
        if secondary_text is not None:
            secondary_text_features = self.module.encode_text(
                secondary_text, normalize=True
            )

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.module.logit_scale.exp(),
            }
            if secondary_text is not None:
                out_dict["secondary_text_features"] = secondary_text_features
            if self.module.logit_bias is not None:
                out_dict["logit_bias"] = self.module.logit_bias
            return out_dict

        out = (image_features, text_features, self.module.logit_scale.exp())
        if secondary_text is not None:
            out += (secondary_text_features,)
        if self.module.logit_bias is not None:
            out += (self.module.logit_bias,)
        return out

    def get_logits(self, image, text):
        return self.module.get_logits(image, text)
