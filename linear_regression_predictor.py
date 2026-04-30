import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_class_weight

data_path = 'GSS_Data_CSV_CodeBook/gss.csv'

# column names from GSS codebook
col_happy = 'GENERAL HAPPINESS'
col_health = 'CONDITION OF HEALTH'
col_relig = 'HOW OFTEN R ATTENDS RELIGIOUS SERVICES'
col_trust = 'CAN PEOPLE BE TRUSTED'
col_finsat = 'SATISFACTION WITH FINANCIAL SITUATION'
col_marital = 'MARITAL STATUS'
col_educ = 'HIGHEST YEAR OF SCHOOL COMPLETED'
col_income = 'TOTAL FAMILY INCOME'
col_excite = 'IS LIFE EXCITING OR DULL'
col_family = 'FAMILY LIFE'
col_friend = 'FRIENDSHIPS'
col_marryhappy = 'HAPPINESS OF MARRIAGE'
col_health_phys = 'HEALTH AND PHYSICAL CONDITION'
col_city = 'CITY OR PLACE R LIVES IN'
col_hobbies = 'NON-WORKING ACTIVITIES,HOBBIES'
col_spouse_educ = 'HIGHEST YEAR SCHOOL COMPLETED, SPOUSE'
col_inc_opinion = 'OPINION OF FAMILY INCOME'
col_job = 'JOB OR HOUSEWORK'
col_class = 'SUBJECTIVE CLASS IDENTIFICATION'
# Bin
attend_map = {1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 4}
FEATURES = ['health', 'religion', 'trust', 'finsat', 'is_married', 'educ', 'income','exciting', 'family_life', 'friendships','marr_happy', 'health_phys', 'city_life', 'hobbies','spouse_educ', 'income_opinion', 'job_sat', 'class_id']
def load_and_clean():
    df = pd.read_csv(data_path, low_memory=False,usecols=[col_happy, col_health, col_relig, col_trust,col_finsat, col_marital, col_educ, col_income,col_excite, col_family, col_friend,col_marryhappy, col_health_phys, col_city,col_hobbies, col_spouse_educ, col_inc_opinion,col_job, col_class])
    df = df.apply(pd.to_numeric, errors='coerce')
    # More cleaning by dropping columns we don't need
    df = df[df[col_happy].isin([1, 2, 3])]
    df = df[df[col_health].isin([1, 2, 3, 4])]
    df = df[df[col_relig].isin(attend_map.keys())]
    df = df[df[col_trust].isin([1, 2, 3])]
    df = df[df[col_finsat].isin([1, 2, 3])]
    df = df[df[col_marital].isin([1, 2, 3, 4, 5])]
    df = df[df[col_educ].between(0, 20)]
    df = df[df[col_income].between(1, 12)]
    # More column cleaning that don't giev me useful info.
    df[col_excite] = df[col_excite].where(df[col_excite].isin([1, 2, 3]))
    df[col_family] = df[col_family].where(df[col_family].between(1, 7))
    df[col_friend] = df[col_friend].where(df[col_friend].between(1, 7))
    df[col_health_phys] = df[col_health_phys].where(df[col_health_phys].between(1, 7))
    df[col_city] = df[col_city].where(df[col_city].between(1, 7))
    df[col_hobbies] = df[col_hobbies].where(df[col_hobbies].between(1, 7))
    df[col_spouse_educ] = df[col_spouse_educ].replace(97, np.nan).where(df[col_spouse_educ].between(0, 20))
    df[col_inc_opinion] = df[col_inc_opinion].where(df[col_inc_opinion].isin([1, 2, 3, 4, 5]))
    df[col_job] = df[col_job].where(df[col_job].isin([1, 2, 3, 4]))
    df[col_class] = df[col_class].where(df[col_class].isin([1, 2, 3, 4]))
    df[col_marryhappy] = df[col_marryhappy].where(df[col_marryhappy].isin([1, 2, 3]))
    # reverse scales so higher = better, if missing insetad of dropping it, we will fill it with median so it doesn't effect the happiense score
    df['happiness'] = 3.5 - df[col_happy]
    df['health'] = 5 - df[col_health]
    df['religion'] = df[col_relig].map(attend_map)
    df['trust'] = 4 - df[col_trust]
    df['finsat'] = 4 - df[col_finsat]
    df['is_married'] = (df[col_marital] == 1).astype(int)
    df['educ'] = df[col_educ]
    df['income'] = df[col_income]
    df['exciting'] = (4 - df[col_excite]).fillna((4 - df[col_excite]).median())
    df['family_life'] = (8 - df[col_family]).fillna((8 - df[col_family]).median())
    df['friendships'] = (8 - df[col_friend]).fillna((8 - df[col_friend]).median())
    df['marr_happy'] = (4 - df[col_marryhappy]).fillna(0)
    df['health_phys'] = (8 - df[col_health_phys]).fillna((8 - df[col_health_phys]).median())
    df['city_life'] = (8 - df[col_city]).fillna((8 - df[col_city]).median())
    df['hobbies'] = (8 - df[col_hobbies]).fillna((8 - df[col_hobbies]).median())
    df['spouse_educ'] = df[col_spouse_educ].fillna(0)
    df['income_opinion'] = (6 - df[col_inc_opinion]).fillna((6 - df[col_inc_opinion]).median())
    df['job_sat'] = (5 - df[col_job]).fillna((5 - df[col_job]).median())
    df['class_id'] = df[col_class].fillna(df[col_class].median())
    return df[['happiness'] + FEATURES]
def train(df):
    X = df[FEATURES]
    y = df['happiness']
    # Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Balance the 3 classes using compute_class
    class_weights = compute_class_weight('balanced', classes=np.array([0.5, 1.5, 2.5]), y=y_train)
    wmap = {0.5: class_weights[0], 1.5: class_weights[1], 2.5: class_weights[2]}
    sample_weights = y_train.map(wmap)
    # LR model
    model = LinearRegression()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_pred)
    # Results
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
    # Graphs
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Linear Regression: Training Diagnostics", fontsize=14)
    groups = [y_pred[y_test == cls] for cls in [0.5, 1.5, 2.5]]
    axes[0].boxplot(groups, tick_labels=["Not Too\nHappy", "Pretty\nHappy", "Very\nHappy"])
    axes[0].set_ylabel("Predicted Happiness Score")
    axes[0].set_title(f"Predictions by Actual Class  (Test RMSE = {mean_squared_error(y_test, y_pred) ** 0.5:.4f})")
    train_sizes, train_scores, val_scores = learning_curve(
        LinearRegression(), X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='neg_root_mean_squared_error')
    axes[1].plot(train_sizes, (-train_scores).mean(axis=1), label='Train RMSE')
    axes[1].plot(train_sizes, (-val_scores).mean(axis=1), label='Validation RMSE')
    axes[1].set_xlabel("Training Samples")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Learning Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    return model
# custom set range
def score_to_label(score):
    if score >= 1.84:
        return "Very Happy"
    elif score >= 1.23:
        return "Pretty Happy"
    else:
        return "Not Too Happy"

# Questionair
def predict(model):
    print("answer these questions:")
    print("\nhow is your health?")
    print("1 = excellent")
    print("2 = good")
    print("3 = fair")
    print("4 = poor")
    health = 5 - int(input("your answer: "))

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
    trust = 4 - int(input("your answer: "))

    print("\nhow satisfied are you with your financial situation?")
    print("1 = satisfied")
    print("2 = more or less")
    print("3 = not at all satisfied")
    finsat = 4 - int(input("your answer: "))

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
    exciting = 4 - int(input("your answer: "))

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

    X_input = pd.DataFrame([[health, religion, trust, finsat, is_married, educ, income, exciting, family_life, friendships, marr_happy, health_phys, city_life, hobbies, spouse_educ, income_opinion, job_sat, class_id]], columns=FEATURES)
    score = model.predict(X_input)[0]
    score_clamped = np.clip(score, 0, 3)
    label = score_to_label(score_clamped)
    print()
    print(f"predicted happiness: {label}")

if __name__ == '__main__':
    print("loading!")
    df = load_and_clean()
    print(f"total samples: {len(df)}\n")
    model = train(df)
    predict(model)
