import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = 'GSS_Data_CSV_CodeBook/gss.csv'

COL_HAPPY  = 'GENERAL HAPPINESS'
COL_HEALTH = 'CONDITION OF HEALTH'
COL_RELIG  = 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES'
COL_TRUST  = 'CAN PEOPLE BE TRUSTED'

# Maps GSS attend codes (1-8) to questionnaire scale (1-4)
# 1=LT ONCE/YR, 2=ONCE/YR -> 1 (Never/Yearly)
# 3=SEVERAL/YR            -> 2 (Once or twice a year)
# 4=ONCE/MO, 5=2-3X/MO   -> 3 (Once a month)
# 6=NRLY/WK, 7=WK, 8=1+/WK -> 4 (Weekly or more)
ATTEND_MAP = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}


def load_and_clean():
    df = pd.read_csv(DATA_PATH, low_memory=False,
                     usecols=[COL_HAPPY, COL_HEALTH, COL_RELIG, COL_TRUST])
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df[df[COL_HAPPY].isin([1, 2, 3])]
    df = df[df[COL_HEALTH].isin([1, 2, 3, 4])]
    df = df[df[COL_RELIG].isin(ATTEND_MAP.keys())]
    df = df[df[COL_TRUST].isin([1, 2, 3])]

    # Reverse happiness and health so higher = better
    df['happiness'] = 4 - df[COL_HAPPY]   # 3=Very Happy, 2=Pretty Happy, 1=Not Too Happy
    df['health']    = 5 - df[COL_HEALTH]  # 4=Excellent, 3=Good, 2=Fair, 1=Poor
    df['religion']  = df[COL_RELIG].map(ATTEND_MAP)
    df['trust']     = 4 - df[COL_TRUST]   # 3=Can Trust, 2=Depends, 1=Can't be too careful

    return df[['happiness', 'health', 'religion', 'trust']]


def train(df):
    X = df[['health', 'religion', 'trust']]
    y = df['happiness']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)

    print("model results:")
    print(f"train r2: {train_r2:.4f}")
    print(f"test r2:  {test_r2:.4f}")
    print(f"mae: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"rmse: {mean_squared_error(y_test, y_pred) ** 0.5:.4f}")
    print("coefficients:")
    for feat, coef in zip(['health', 'religion', 'trust'], model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Regression: Training vs Test Accuracy", fontsize=14)

    for ax, y_true, y_pred_vals, label, r2 in [
        (axes[0], y_train, y_train_pred, "Train", train_r2),
        (axes[1], y_test, y_pred, "Test", test_r2),
    ]:
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(y_true))
        ax.scatter(np.array(y_true) + jitter, y_pred_vals, alpha=0.3, edgecolors='none', s=20)
        ax.plot([1, 3], [1, 3], 'r--', linewidth=1.5)
        ax.set_title(f"{label} Set  (R² = {r2:.4f})")
        ax.set_xlabel("Actual Happiness")
        ax.set_ylabel("Predicted Happiness")
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0.5, 3.5)
        ax.set_xticks([1, 2, 3])
        ax.set_yticks([1, 2, 3])
        ax.set_xticklabels(["Not Too\nHappy", "Pretty\nHappy", "Very\nHappy"])
        ax.set_yticklabels(["Not Too\nHappy", "Pretty\nHappy", "Very\nHappy"])

    plt.tight_layout()
    plt.show()

    return model


def score_to_label(score):
    if score >= 2.5:
        return "Very Happy"
    elif score >= 1.5:
        return "Pretty Happy"
    else:
        return "Not Too Happy"


def predict(model):
    print("answer these questions:")

    print("\nhow is your health?")
    print("1 = excellent")
    print("2 = good")
    print("3 = fair")
    print("4 = poor")
    health_raw = int(input("your answer: "))
    health = 5 - health_raw

    print("\nhow often do you go to religious services?")
    print("1 = never or yearly")
    print("2 = once or twice a year")
    print("3 = once a month")
    print("4 = weekly or more")
    religion = int(input("your answer: "))

    print("\ndo you think most people can be trusted?")
    print("1 = yes can be trusted")
    print("2 = no cant be too careful")
    print("3 = depends")
    trust_raw = int(input("your answer: "))
    trust = 4 - trust_raw

    X_input = pd.DataFrame([[health, religion, trust]],
                           columns=['health', 'religion', 'trust'])
    score = model.predict(X_input)[0]
    score_clamped = np.clip(score, 1, 3)
    label = score_to_label(score_clamped)

    print()
    print(f"predicted happiness: {label}")


if __name__ == '__main__':
    print("loading data...")
    df = load_and_clean()
    print(f"total samples: {len(df)}\n")

    model = train(df)
    predict(model)
