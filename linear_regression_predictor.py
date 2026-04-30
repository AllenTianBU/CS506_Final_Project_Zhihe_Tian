import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = 'GSS_Data_CSV_CodeBook/gss.csv'

COL_HAPPY   = 'GENERAL HAPPINESS'
COL_HEALTH  = 'CONDITION OF HEALTH'
COL_RELIG   = 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES'
COL_TRUST   = 'CAN PEOPLE BE TRUSTED'
COL_FINSAT  = 'SATISFACTION WITH FINANCIAL SITUATION'
COL_MARITAL = 'MARITAL STATUS'
COL_EDUC    = 'HIGHEST YEAR OF SCHOOL COMPLETED'
COL_INCOME  = 'TOTAL FAMILY INCOME'
COL_EXCITE      = 'IS LIFE EXCITING OR DULL'
COL_FAMILY      = 'FAMILY LIFE'
COL_FRIEND      = 'FRIENDSHIPS'
COL_MARRYHAPPY  = 'HAPPINESS OF MARRIAGE'
COL_HEALTH_PHYS = 'HEALTH AND PHYSICAL CONDITION'
COL_CITY        = 'CITY OR PLACE R LIVES IN'
COL_HOBBIES     = 'NON-WORKING ACTIVITIES,HOBBIES'
COL_SPOUSE_EDUC = 'HIGHEST YEAR SCHOOL COMPLETED, SPOUSE'
COL_INC_OPINION = 'OPINION OF FAMILY INCOME'
COL_JOB         = 'JOB OR HOUSEWORK'
COL_CLASS       = 'SUBJECTIVE CLASS IDENTIFICATION'
ATTEND_MAP = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}

FEATURES = ['health', 'religion', 'trust', 'finsat', 'is_married', 'educ', 'income',
            'exciting', 'family_life', 'friendships',
            'marr_happy', 'health_phys', 'city_life', 'hobbies',
            'spouse_educ', 'income_opinion', 'job_sat', 'class_id']


def load_and_clean():
    df = pd.read_csv(DATA_PATH, low_memory=False,
                     usecols=[COL_HAPPY, COL_HEALTH, COL_RELIG, COL_TRUST,
                               COL_FINSAT, COL_MARITAL, COL_EDUC, COL_INCOME,
                               COL_EXCITE, COL_FAMILY, COL_FRIEND,
                               COL_MARRYHAPPY, COL_HEALTH_PHYS, COL_CITY,
                               COL_HOBBIES, COL_SPOUSE_EDUC, COL_INC_OPINION,
                               COL_JOB, COL_CLASS])
    df = df.apply(pd.to_numeric, errors='coerce')

    df = df[df[COL_HAPPY].isin([1, 2, 3])]
    df = df[df[COL_HEALTH].isin([1, 2, 3, 4])]
    df = df[df[COL_RELIG].isin(ATTEND_MAP.keys())]
    df = df[df[COL_TRUST].isin([1, 2, 3])]
    df = df[df[COL_FINSAT].isin([1, 2, 3])]
    df = df[df[COL_MARITAL].isin([1, 2, 3, 4, 5])]
    df = df[df[COL_EDUC].between(0, 20)]
    df = df[df[COL_INCOME].between(1, 12)]

    df[COL_EXCITE] = df[COL_EXCITE].where(df[COL_EXCITE].isin([1, 2, 3]))
    df[COL_FAMILY] = df[COL_FAMILY].where(df[COL_FAMILY].between(1, 7))
    df[COL_FRIEND] = df[COL_FRIEND].where(df[COL_FRIEND].between(1, 7))
    df[COL_HEALTH_PHYS] = df[COL_HEALTH_PHYS].where(df[COL_HEALTH_PHYS].between(1, 7))
    df[COL_CITY] = df[COL_CITY].where(df[COL_CITY].between(1, 7))
    df[COL_HOBBIES] = df[COL_HOBBIES].where(df[COL_HOBBIES].between(1, 7))
    df[COL_SPOUSE_EDUC] = df[COL_SPOUSE_EDUC].replace(97, np.nan).where(df[COL_SPOUSE_EDUC].between(0, 20))
    df[COL_INC_OPINION] = df[COL_INC_OPINION].where(df[COL_INC_OPINION].isin([1, 2, 3, 4, 5]))
    df[COL_JOB] = df[COL_JOB].where(df[COL_JOB].isin([1, 2, 3, 4]))
    df[COL_CLASS] = df[COL_CLASS].where(df[COL_CLASS].isin([1, 2, 3, 4]))
    df[COL_MARRYHAPPY] = df[COL_MARRYHAPPY].where(df[COL_MARRYHAPPY].isin([1, 2, 3]))
    df['happiness']      = 3.5 - df[COL_HAPPY]
    df['health']         = 5 - df[COL_HEALTH]
    df['religion']       = df[COL_RELIG].map(ATTEND_MAP)
    df['trust']          = 4 - df[COL_TRUST]
    df['finsat']         = 4 - df[COL_FINSAT]
    df['is_married']     = (df[COL_MARITAL] == 1).astype(int)
    df['educ']           = df[COL_EDUC]
    df['income']         = df[COL_INCOME]
    df['exciting']       = (4 - df[COL_EXCITE]).fillna((4 - df[COL_EXCITE]).median())
    df['family_life']    = (8 - df[COL_FAMILY]).fillna((8 - df[COL_FAMILY]).median())
    df['friendships']    = (8 - df[COL_FRIEND]).fillna((8 - df[COL_FRIEND]).median())
    df['marr_happy']     = (4 - df[COL_MARRYHAPPY]).fillna(0)
    df['health_phys']    = (8 - df[COL_HEALTH_PHYS]).fillna((8 - df[COL_HEALTH_PHYS]).median())
    df['city_life']      = (8 - df[COL_CITY]).fillna((8 - df[COL_CITY]).median())
    df['hobbies']        = (8 - df[COL_HOBBIES]).fillna((8 - df[COL_HOBBIES]).median())
    df['spouse_educ']    = df[COL_SPOUSE_EDUC].fillna(0)
    df['income_opinion'] = (6 - df[COL_INC_OPINION]).fillna((6 - df[COL_INC_OPINION]).median())
    df['job_sat']        = (5 - df[COL_JOB]).fillna((5 - df[COL_JOB]).median())
    df['class_id']       = df[COL_CLASS].fillna(df[COL_CLASS].median())

    return df[['happiness'] + FEATURES]


def train(df):
    X = df[FEATURES]
    y = df['happiness']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    class_weights = compute_class_weight('balanced', classes=np.array([0.5, 1.5, 2.5]), y=y_train)
    weight_map = {0.5: class_weights[0], 1.5: class_weights[1], 2.5: class_weights[2]}
    sample_weights = y_train.map(weight_map)

    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=sample_weights)

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
    for feat, coef in zip(FEATURES, model.coef_):
        print(f"  {feat}: {coef:.4f}")
    print(f"  intercept: {model.intercept_:.4f}")
    print()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Regression: Training Diagnostics", fontsize=14)

    groups = [y_pred[y_test == cls] for cls in [0.5, 1.5, 2.5]]
    axes[0].boxplot(groups, tick_labels=["Not Too\nHappy", "Pretty\nHappy", "Very\nHappy"])
    axes[0].set_ylabel("Predicted Happiness Score")
    axes[0].set_title(f"Predictions by Actual Class  (Test RMSE = {mean_squared_error(y_test, y_pred) ** 0.5:.4f})")

    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(), X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_root_mean_squared_error')
    train_rmse = -train_scores
    val_rmse = -val_scores
    axes[1].plot(train_sizes, train_rmse.mean(axis=1), label='Train RMSE')
    axes[1].plot(train_sizes, val_rmse.mean(axis=1), label='Validation RMSE')
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Learning Curve")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return model


def score_to_label(score):
    if score >= 1.84:
        return "Very Happy"
    elif score >= 1.23:
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

    print("\nhow satisfied are you with your financial situation?")
    print("1 = satisfied")
    print("2 = more or less")
    print("3 = not at all satisfied")
    finsat_raw = int(input("your answer: "))
    finsat = 4 - finsat_raw

    print("\nare you currently married?")
    print("1 = yes")
    print("2 = no")
    is_married = 1 if int(input("your answer: ")) == 1 else 0

    print("\nhow many years of school have you completed? (0-20)")
    educ = int(input("your answer: "))

    print("\nwhat is your total family income level?")
    print("1 = under $1,000   2 = $1,000-$2,999   3 = $3,000-$3,999")
    print("4 = $4,000-$4,999  5 = $5,000-$5,999   6 = $6,000-$6,999")
    print("7 = $7,000-$7,999  8 = $8,000-$9,999   9 = $10,000-$12,499")
    print("10 = $12,500-$14,999  11 = $15,000-$17,499  12 = $17,500-$19,999")
    income = int(input("your answer: "))

    print("\nhow exciting is your life in general?")
    print("1 = exciting")
    print("2 = routine")
    print("3 = dull")
    exciting_raw = int(input("your answer: "))
    exciting = 4 - exciting_raw

    print("\nhow satisfied are you with your family life?")
    print("1 = completely satisfied   7 = completely dissatisfied")
    family_life = 8 - int(input("your answer (1-7): "))

    print("\nhow satisfied are you with your friendships?")
    print("1 = completely satisfied   7 = completely dissatisfied")
    friendships = 8 - int(input("your answer (1-7): "))

    if is_married:
        print("\nhow happy is your marriage?")
        print("1 = very happy")
        print("2 = pretty happy")
        print("3 = not too happy")
        marr_happy = 4 - int(input("your answer: "))
    else:
        marr_happy = 0

    print("\nhow satisfied are you with your health and physical condition?")
    print("1 = completely satisfied   7 = completely dissatisfied")
    health_phys = 8 - int(input("your answer (1-7): "))

    print("\nhow satisfied are you with the city or place you live in?")
    print("1 = completely satisfied   7 = completely dissatisfied")
    city_life = 8 - int(input("your answer (1-7): "))

    print("\nhow satisfied are you with your hobbies and non-working activities?")
    print("1 = completely satisfied   7 = completely dissatisfied")
    hobbies = 8 - int(input("your answer (1-7): "))

    print("\nhow many years of school has your spouse completed? (0 if no spouse)")
    spouse_educ = int(input("your answer: "))

    print("\ncompared to others, how would you describe your family income?")
    print("1 = far above average   2 = above average   3 = average")
    print("4 = below average       5 = far below average")
    income_opinion = 6 - int(input("your answer: "))

    print("\nhow satisfied are you with your job or housework?")
    print("1 = very satisfied   2 = moderately satisfied")
    print("3 = a little dissatisfied   4 = very dissatisfied")
    job_sat = 5 - int(input("your answer: "))

    print("\nwhich social class do you identify with?")
    print("1 = lower class   2 = working class   3 = middle class   4 = upper class")
    class_id = int(input("your answer: "))

    X_input = pd.DataFrame([[health, religion, trust, finsat, is_married, educ, income,
                             exciting, family_life, friendships,
                             marr_happy, health_phys, city_life, hobbies,
                             spouse_educ, income_opinion, job_sat, class_id]],
                           columns=FEATURES)
    score = model.predict(X_input)[0]
    score_clamped = np.clip(score, 0, 3)
    label = score_to_label(score_clamped)

    print()
    print(f"predicted happiness: {label}")


if __name__ == '__main__':
    print("loading data...")
    df = load_and_clean()
    print(f"total samples: {len(df)}\n")

    model = train(df)
    #predict(model)
