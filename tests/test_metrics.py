from src.training.metrics import calculate_metrics

y_true = [0,1,1,0,1,0]

y_pred = [0,1,0,0,1,1]

y_prob = [

    0.12,

    0.91,

    0.45,

    0.18,

    0.83,

    0.63

]

metrics = calculate_metrics(

    y_true,

    y_pred,

    y_prob

)

for k,v in metrics.items():

    print(f"{k:12} {v:.4f}")