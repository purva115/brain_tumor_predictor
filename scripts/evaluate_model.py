import os
import sys
import json
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.dataloader import BrainMRIDataset
from src.models.cnn_classifier import BrainMRIClassifier


class ModelEvaluator:
    console = Console()
    """Comprehensive model evaluation with Tinker API integration"""

    def __init__(self, model_path, dataset_path, use_tinker=False):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.use_tinker = use_tinker
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = BrainMRIClassifier().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load dataset
        self.dataset = BrainMRIDataset(dataset_path)
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        _, self.test_set = random_split(
            self.dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        self.test_loader = DataLoader(self.test_set, batch_size=16, shuffle=False)

        # Initialize Tinker client if requested
        self.tinker_client = None
        if use_tinker:
            self._initialize_tinker()

    def _initialize_tinker(self):
        """Initialize Tinker API client"""
        api_key = os.getenv('TINKER_API_KEY') or os.getenv('TINKER_KEY')
        if not api_key:
            self.console.print(Panel(
                Text("TINKER_API_KEY not found. Tinker integration disabled.\nSet TINKER_API_KEY environment variable to enable Tinker features.", justify="left"),
                title="[yellow]Warning[/yellow]",
                border_style="yellow"
            ))
            self.use_tinker = False
            return

        try:
            import tinker
            self.tinker_client = tinker.ServiceClient()
            self.console.print(Panel("Tinker API initialized successfully", title="[green]Success[/green]", border_style="green"))
        except ImportError:
            self.console.print(Panel(
                Text("'tinker' package not installed. Run: pip install tinker", justify="left"),
                title="[yellow]Warning[/yellow]",
                border_style="yellow"
            ))
            self.use_tinker = False
        except Exception as e:
            self.console.print(Panel(
                Text(f"Failed to initialize Tinker client: {e}", justify="left"),
                title="[yellow]Warning[/yellow]",
                border_style="yellow"
            ))
            self.use_tinker = False

    def get_predictions(self):
        """Get model predictions and true labels"""
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def calculate_metrics(self, y_true, y_pred, y_probs):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }

        # Calculate per-class metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics['confusion_matrix'] = {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        }

        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, save_path='assets/confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Tumor', 'Tumor'],
                    yticklabels=['No Tumor', 'Tumor'])
        plt.title('Confusion Matrix - Brain Tumor Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.console.print(Panel(f"Confusion matrix saved to {save_path}", title="[green]Success[/green]", border_style="green"))

    def plot_prediction_examples(self, y_true, y_pred, save_path='assets/prediction_examples.png', num_examples=5):
        """Plot random examples of correct and incorrect predictions"""
        # Collect indices
        correct_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
        incorrect_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]

        # Limit to num_examples
        correct_sample = random.sample(correct_indices, min(len(correct_indices), num_examples))
        incorrect_sample = random.sample(incorrect_indices, min(len(incorrect_indices), num_examples))

        plt.figure(figsize=(15, 6))

        # Helper to plot
        def plot_images(indices, row_idx, title_prefix):
            for i, idx in enumerate(indices):
                img_tensor, _ = self.test_set[idx]
                # Denormalize
                img = img_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
                img = np.clip(img, 0, 1)

                pred_label = y_pred[idx]
                true_label = y_true[idx]

                class_names = ['No Tumor', 'Tumor']

                plt.subplot(2, num_examples, row_idx * num_examples + i + 1)
                plt.imshow(img)
                plt.title(f"{title_prefix}\nTrue: {class_names[true_label]}\nPred: {class_names[pred_label]}",
                         color='green' if true_label == pred_label else 'red', fontsize=8)
                plt.axis('off')

        if correct_sample:
            plot_images(correct_sample, 0, "Correct")
        
        if incorrect_sample:
            plot_images(incorrect_sample, 1, "Incorrect")
        else:
            # Handle perfect accuracy case
            plt.figtext(0.5, 0.25, "No incorrect predictions (Accuracy 100%)", 
                       ha='center', va='center', fontsize=12)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.console.print(Panel(f"Prediction examples saved to {save_path}", title="[green]Success[/green]", border_style="green"))

    def analyze_model_complexity(self):
        """Analyze current model complexity"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            'model_architecture': 'ResNet18',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': os.path.getsize(self.model_path) / (1024 * 1024)
        }

    def recommend_improvements(self, metrics):
        """Provide recommendations based on performance metrics"""
        recommendations = []
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1_score']

        # Overall performance assessment
        if accuracy < 0.85:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Low overall accuracy',
                'current_value': f'{accuracy:.3f}',
                'recommendation': 'Consider upgrading to a more sophisticated model architecture',
                'suggestions': [
                    'ResNet50 or ResNet101 for deeper feature extraction',
                    'DenseNet121/169 for better gradient flow',
                    'EfficientNet-B0 to B7 for optimal accuracy-efficiency trade-off',
                    'Vision Transformer (ViT) for state-of-the-art performance'
                ]
            })
        elif accuracy < 0.90:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': 'Moderate accuracy - room for improvement',
                'current_value': f'{accuracy:.3f}',
                'recommendation': 'Consider model enhancements or ensemble methods',
                'suggestions': [
                    'Fine-tune with larger medical imaging dataset (e.g., transfer learning from medical pretrained models)',
                    'Implement ensemble methods (combining multiple models)',
                    'Try ResNet34 or EfficientNet-B1 as incremental upgrades'
                ]
            })
        else:
            recommendations.append({
                'priority': 'LOW',
                'issue': 'Good overall accuracy',
                'current_value': f'{accuracy:.3f}',
                'recommendation': 'Current model performs well, but fine-tuning may help',
                'suggestions': [
                    'Data augmentation to improve generalization',
                    'Hyperparameter optimization',
                    'Focus on reducing false negatives/positives'
                ]
            })

        # Check for class imbalance issues
        if abs(precision - recall) > 0.15:
            recommendations.append({
                'priority': 'HIGH',
                'issue': 'Precision-Recall imbalance detected',
                'current_value': f'Precision: {precision:.3f}, Recall: {recall:.3f}',
                'recommendation': 'Address class imbalance',
                'suggestions': [
                    'Implement class weights in loss function',
                    'Use focal loss for hard examples',
                    'Apply data augmentation to minority class',
                    'Consider SMOTE or other synthetic data generation'
                ]
            })

        # Medical imaging specific recommendations
        recommendations.append({
            'priority': 'INFO',
            'issue': 'Medical imaging best practices',
            'recommendation': 'Consider domain-specific improvements',
            'suggestions': [
                'Use pretrained models from medical imaging datasets (e.g., RadImageNet)',
                'Implement grad-CAM or attention maps for interpretability',
                'Add uncertainty quantification for clinical reliability',
                'Consider 3D CNNs if volumetric MRI data is available',
                'Validate with external datasets for generalization assessment'
            ]
        })

        return recommendations

    def save_evaluation_report(self, metrics, complexity, recommendations,
                              save_path='evaluation_results/evaluation_report.json'):
        """Save comprehensive evaluation report"""
        report = {
            'model_path': str(self.model_path),
            'dataset_info': {
                'total_samples': len(self.dataset),
                'test_samples': len(self.test_set),
                'train_samples': len(self.dataset) - len(self.test_set)
            },
            'performance_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall']),
                'f1_score': float(metrics['f1_score']),
                'sensitivity': float(metrics['sensitivity']),
                'specificity': float(metrics['specificity'])
            },
            'confusion_matrix': metrics['confusion_matrix'],
            'model_complexity': complexity,
            'recommendations': recommendations,
            'evaluation_date': str(np.datetime64('now'))
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        self.console.print(Panel(f"Evaluation report saved to {save_path}", title="[green]Success[/green]", border_style="green"))
        return report

    def print_summary(self, metrics, recommendations):
        """Print evaluation summary to console"""
        self.console.print(Panel(
            Text("BRAIN TUMOR DETECTION MODEL EVALUATION REPORT", justify="center"),
            title="[bold cyan]Evaluation Report[/bold cyan]",
            border_style="cyan"
        ))

        metrics_text = Text()
        metrics_text.append(f"Accuracy:     {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.2f}%)\n")
        metrics_text.append(f"Precision:    {metrics['precision']:.3f}\n")
        metrics_text.append(f"Recall:       {metrics['recall']:.3f}\n")
        metrics_text.append(f"F1 Score:     {metrics['f1_score']:.3f}\n")
        metrics_text.append(f"Sensitivity:  {metrics['sensitivity']:.3f}\n")
        metrics_text.append(f"Specificity:  {metrics['specificity']:.3f}\n")
        self.console.print(Panel(metrics_text, title="[bold green]Performance Metrics[/bold green]", border_style="green"))

        cm = metrics['confusion_matrix']
        cm_text = Text()
        cm_text.append(f"True Negatives:  {cm['true_negative']}\n")
        cm_text.append(f"False Positives: {cm['false_positive']}\n")
        cm_text.append(f"False Negatives: {cm['false_negative']}\n")
        cm_text.append(f"True Positives:  {cm['true_positive']}\n")
        self.console.print(Panel(cm_text, title="[bold blue]Confusion Matrix[/bold blue]", border_style="blue"))

        for i, rec in enumerate(recommendations, 1):
            rec_text = Text()
            if 'current_value' in rec:
                rec_text.append(f"Current: {rec['current_value']}\n")
            rec_text.append(f"Recommendation: {rec['recommendation']}\n")
            if rec['suggestions']:
                rec_text.append("Suggestions:\n")
                for sug in rec['suggestions'][:3]:
                    rec_text.append(f"  • {sug}\n")
            self.console.print(Panel(rec_text, title=f"[bold yellow]Recommendation {i}: {rec['issue']} ({rec['priority']})[/bold yellow]", border_style="yellow"))

        if metrics['accuracy'] >= 0.90:
            verdict_text = "Current model performs well for this task.\nA more sophisticated model may provide marginal improvements."
            verdict_title = "[bold green]Verdict[/bold green]"
            verdict_style = "green"
        elif metrics['accuracy'] >= 0.85:
            verdict_text = "Model shows decent performance.\nConsider incremental improvements or ensemble methods."
            verdict_title = "[bold yellow]Verdict[/bold yellow]"
            verdict_style = "yellow"
        else:
            verdict_text = "Model performance is suboptimal.\nA more sophisticated model architecture is RECOMMENDED."
            verdict_title = "[bold red]Verdict[/bold red]"
            verdict_style = "red"
        self.console.print(Panel(Text(verdict_text, justify="center"), title=verdict_title, border_style=verdict_style))

    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        self.console.print(Panel(
            f"Device: {self.device}\nModel: {self.model_path}\nDataset: {self.dataset_path}\nTest samples: {len(self.test_set)}",
            title="[bold cyan]Starting Model Evaluation[/bold cyan]",
            border_style="cyan"
        ))

        with self.console.status("[bold green]Running evaluation...[/bold green]") as status:
            status.update("Generating predictions...")
            y_true, y_pred, y_probs = self.get_predictions()

            status.update("Calculating metrics...")
            metrics = self.calculate_metrics(y_true, y_pred, y_probs)

            status.update("Generating visualizations...")
            self.plot_confusion_matrix(y_true, y_pred)
            self.plot_prediction_examples(y_true, y_pred)

            status.update("Analyzing model complexity...")
            complexity = self.analyze_model_complexity()

            status.update("Generating recommendations...")
            recommendations = self.recommend_improvements(metrics)

            status.update("Saving evaluation report...")
            self.save_evaluation_report(metrics, complexity, recommendations)

        self.print_summary(metrics, recommendations)

        return metrics, recommendations


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate brain tumor detection model performance'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/brain_tumor_model.pth',
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/BrainMRI',
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--use-tinker',
        action='store_true',
        help='Enable Tinker API integration for advanced analysis'
    )

    args = parser.parse_args()

    console = Console()
    # Check if model exists
    if not os.path.exists(args.model_path):
        console.print(Panel(
            Text(f"Model file not found at {args.model_path}\nPlease train the model first using: python -m src.training.train_model", justify="left"),
            title="[red]Error[/red]",
            border_style="red"
        ))
        sys.exit(1)

    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        console.print(Panel(
            Text(f"Dataset not found at {args.dataset_path}", justify="left"),
            title="[red]Error[/red]",
            border_style="red"
        ))
        sys.exit(1)

    # Run evaluation
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        use_tinker=args.use_tinker
    )

    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
