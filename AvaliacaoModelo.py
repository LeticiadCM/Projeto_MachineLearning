import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, RocCurveDisplay, auc, PrecisionRecallDisplay
from sklearn.model_selection import StratifiedKFold

def evaluate_model(model, pad_train, labels_train):
    
    wrapped_model = KerasClassifier(model=model, epochs=5, verbose=0)
    
    # (ref. https://colab.research.google.com/drive/15oCn3s3wRkra87rO727gFMKGj3HgVZWr?usp=sharing#scrollTo=Nn4Izbr8FMcq)
    X = pad_train 
    y = labels_train
    n_splits = 6  # Número de divisões para validação cruzada
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Configuração do KFold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Variáveis para ROC
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Variáveis para PR
    precisions = []
    recalls = []

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 6))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Dividir os dados em treino e teste
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Ajustar o modelo
        wrapped_model.fit(X_train, y_train)

        # Curva ROC
        viz_roc = RocCurveDisplay.from_estimator(
            wrapped_model,
            X_test,
            y_test,
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax_roc,
        )
        
        interp_tpr = np.interp(mean_fpr, viz_roc.fpr, viz_roc.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz_roc.roc_auc)

        # Curva PR
        viz_pr = PrecisionRecallDisplay.from_estimator(
            wrapped_model,
            X_test,
            y_test,
            name=f"PR fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax_pr,
        )
        precisions.append(viz_pr.precision)
        recalls.append(viz_pr.recall)

    # ROC Média
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax_roc.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Curva ROC média (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    ax_roc.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - np.std(tprs, axis=0), 0),
        np.minimum(mean_tpr + np.std(tprs, axis=0), 1),
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax_roc.set(title="Curva ROC Média", xlabel="Taxa de Falsos Positivos", ylabel="Taxa de Verdadeiros Positivos")
    ax_roc.legend(loc="lower right")

    # PR Média
    ax_pr.set(title="Curva Precision-Recall Média", xlabel="Recall", ylabel="Precision")
    ax_pr.legend(loc="lower left")

    # Mostrar e salvar gráficos
    plt.tight_layout()
    plt.savefig(f"roc_pr_curves_{dt}.png", dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.close()
       
    # Exportar relatório
    train_pred = model.predict(pad_train) > 0.5
    report = classification_report(labels_train, train_pred, target_names = ["Negativo", "Positivo"], output_dict = True)
    report_df = pd.DataFrame(report).transpose()
    
    report_df.to_csv(f"classification_report{dt}.csv", index=True)
    print(f"Relatório de classificação exportado como classification_report{dt}.csv")
    
    # Exportar gráficos
    fig.savefig(f"roc_pr_curves_{dt}.png", dpi=300, bbox_inches="tight")
    print(f"Gráficos exportados como 'roc_pr_curves{dt}.png'.")
    plt.show()


 