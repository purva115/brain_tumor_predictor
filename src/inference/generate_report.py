def generate_report(pred_class, probs):
    conditions = ['Normal', 'Tumor']
    confidence = round(max(probs) * 100, 2)
    condition = conditions[pred_class]
    report = f"MRI scan indicates **{condition}** condition with **{confidence}%** confidence."
    return report
