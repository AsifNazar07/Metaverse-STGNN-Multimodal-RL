import os
import nltk
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dataset_loader import VOC2012CaptionDataset


class CaptionEvaluator:

    def __init__(self, model_path, image_dir, annotation_file):
        print("[INFO] Loading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()

        self.dataset = VOC2012CaptionDataset(
            image_dir=image_dir,
            annotation_file=annotation_file,
            processor=self.processor
        )

        self.rouge = Rouge()

    # ----------------------------------------------------------------------
    # Generate caption 
    # ----------------------------------------------------------------------
    def generate_caption(self, img_path):
        img = Image.open(img_path).convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt")
        output = self.model.generate(**inputs)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    # ----------------------------------------------------------------------
    # Evaluate full dataset
    # ----------------------------------------------------------------------
    def evaluate_dataset(self, num_samples=100):
        bleu_scores = []
        rouge_scores = []

        print(f"[INFO] Evaluating {num_samples} samples...")

        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]

            input_ids = sample["input_ids"].unsqueeze(0)
            pixel_values = sample["pixel_values"].unsqueeze(0)

            output = self.model.generate(
                input_ids=None,
                pixel_values=pixel_values
            )
            pred_caption = self.processor.decode(output[0], skip_special_tokens=True)

            true_caption = self.processor.decode(
                sample["labels"],
                skip_special_tokens=True
            )

            # BLEU
            bleu = sentence_bleu([true_caption.split()], pred_caption.split())
            bleu_scores.append(bleu)

            # ROUGE-L
            rouge = self.rouge.get_scores(pred_caption, true_caption)[0]["rouge-l"]["f"]
            rouge_scores.append(rouge)

        print("\n[RESULTS] BLIP Fine-Tuned Model Evaluation")
        print("-------------------------------------------------")
        print(f"BLEU Score   : {sum(bleu_scores)/len(bleu_scores):.4f}")
        print(f"ROUGE-L Score: {sum(rouge_scores)/len(rouge_scores):.4f}")

        return bleu_scores, rouge_scores


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":

    model_path = "./EBLIP_VOC2012_Finetuned"
    image_dir = "PATH/TO/VOC2012/JPEGImages"
    annotation_file = "PATH/TO/VOC2012/Annotations.json"

    evaluator = CaptionEvaluator(model_path, image_dir, annotation_file)

    # Test a single caption
    test_img = os.path.join(image_dir, "2008_000008.jpg")
    print("\n[Test Caption]")
    print("Generated:", evaluator.generate_caption(test_img))

    # Dataset evaluation
    evaluator.evaluate_dataset(num_samples=50)
