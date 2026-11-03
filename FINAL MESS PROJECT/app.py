import streamlit as st
import sqlite3
import os
from datetime import date
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# 1. APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Mess Food Demand Predictor",
    page_icon="🍱",
    layout="wide"
)

DB_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mess_data.db")


# ============================================================
# 2. DATABASE FUNCTIONS
# ============================================================

def get_connection():
    return sqlite3.connect(
        DB_NAME,
        check_same_thread=False
    )


def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    # Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_date TEXT NOT NULL,
            meal TEXT NOT NULL,
            food_type TEXT NOT NULL,
            menu TEXT NOT NULL,
            total_students INTEGER NOT NULL,
            actual_diners INTEGER NOT NULL,
            exam INTEGER DEFAULT 0,
            holiday INTEGER DEFAULT 0,
            event INTEGER DEFAULT 0,
            popularity INTEGER DEFAULT 3,
            notes TEXT
        )
    """)

    # Food item library
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_name TEXT UNIQUE NOT NULL,
            category TEXT,
            portion REAL NOT NULL,
            unit TEXT NOT NULL
        )
    """)

    # Food waste table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS food_waste (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_date TEXT NOT NULL,
            meal TEXT NOT NULL,
            item_name TEXT NOT NULL,
            prepared_qty REAL NOT NULL,
            leftover_qty REAL NOT NULL,
            unit TEXT NOT NULL,
            cost_per_unit REAL DEFAULT 0
        )
    """)

    # Prediction history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            meal TEXT NOT NULL,
            food_type TEXT NOT NULL,
            expected_students INTEGER NOT NULL,
            predicted_diners INTEGER NOT NULL,
            model TEXT NOT NULL,
            confidence TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def execute_query(query, params=()):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        query,
        params
    )

    conn.commit()
    conn.close()


def get_dataframe(query, params=()):

    conn = get_connection()

    df = pd.read_sql_query(
        query,
        conn,
        params=params
    )

    conn.close()

    return df


initialize_database()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def add_display_id(df):

    result = df.copy()

    result.insert(
        0,
        "display_id",
        range(1, len(result) + 1)
    )

    return result


# ============================================================
# 4. AI PREDICTION FUNCTION
# ============================================================

def predict_diners(
    pred_meal,
    pred_food_type,
    pred_students,
    pred_exam,
    pred_holiday,
    pred_event,
    pred_popularity
):

    history = get_dataframe("""
        SELECT
            meal,
            food_type,
            total_students,
            actual_diners,
            exam,
            holiday,
            event,
            popularity
        FROM attendance
    """)

    record_count = len(history)


    # --------------------------------------------------------
    # No historical data
    # --------------------------------------------------------

    if record_count == 0:

        ratio = 0.70

        if pred_exam:
            ratio *= 0.75

        if pred_holiday:
            ratio *= 0.60

        if pred_event:
            ratio *= 1.05

        ratio *= (
            1 + ((pred_popularity - 3) * 0.04)
        )

        prediction = pred_students * ratio

        prediction = max(
            0,
            min(prediction, pred_students)
        )

        return (
            int(round(prediction)),
            "Rule-Based Initial Estimate",
            "Low / No Historical Data"
        )


    # --------------------------------------------------------
    # Early data: less than 12 records
    # --------------------------------------------------------

    if record_count < 12:

        ratios = (
            history["actual_diners"]
            / history["total_students"].replace(0, np.nan)
        )

        ratios = ratios.dropna()

        if ratios.empty:
            ratio = 0.70
        else:
            ratio = float(ratios.mean())


        if pred_exam:
            ratio *= 0.75

        if pred_holiday:
            ratio *= 0.60

        if pred_event:
            ratio *= 1.05

        ratio *= (
            1 + ((pred_popularity - 3) * 0.04)
        )


        prediction = pred_students * ratio

        prediction = max(
            0,
            min(prediction, pred_students)
        )

        return (
            int(round(prediction)),
            "Historical + Rule-Based Estimate",
            "Low / Early Data"
        )


    # --------------------------------------------------------
    # Machine Learning section
    # --------------------------------------------------------

    X = history[
        [
            "meal",
            "food_type",
            "total_students",
            "exam",
            "holiday",
            "event",
            "popularity"
        ]
    ]

    y = history["actual_diners"]


    categorical_features = [
        "meal",
        "food_type"
    ]

    numeric_features = [
        "total_students",
        "exam",
        "holiday",
        "event",
        "popularity"
    ]


    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_features
            ),
            (
                "numeric",
                "passthrough",
                numeric_features
            )
        ]
    )


    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=180,
                    max_depth=8,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )


    model.fit(
        X,
        y
    )


    prediction_input = pd.DataFrame([
        {
            "meal": pred_meal,
            "food_type": pred_food_type,
            "total_students": pred_students,
            "exam": int(pred_exam),
            "holiday": int(pred_holiday),
            "event": int(pred_event),
            "popularity": pred_popularity
        }
    ])


    ml_prediction = float(
        model.predict(prediction_input)[0]
    )


    # Historical attendance ratio
    ratios = (
        history["actual_diners"]
        / history["total_students"].replace(0, np.nan)
    )

    ratios = ratios.dropna()


    if ratios.empty:
        historical_ratio = 0.70
    else:
        historical_ratio = float(
            ratios.mean()
        )


    historical_prediction = (
        pred_students
        * historical_ratio
    )


    # 75% ML + 25% historical
    prediction = (
        0.75 * ml_prediction
        + 0.25 * historical_prediction
    )


    # Small context adjustments
    if pred_exam:
        prediction *= 0.95

    if pred_holiday:
        prediction *= 0.90

    if pred_event:
        prediction *= 1.03


    # Never predict more than expected students
    prediction = max(
        0,
        min(prediction, pred_students)
    )


    if record_count < 30:
        confidence = "Medium"
    else:
        confidence = "Higher"


    return (
        int(round(prediction)),
        "Random Forest + Historical Blend",
        confidence
    )


# ============================================================
# 5. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🍱 AI Mess Project")

st.sidebar.caption(
    "AI Mess Food Demand Predictor"
)


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📝 Add Attendance",
        "🤖 AI Demand Prediction",
        "🍱 Quantity Calculator",
        "📚 Food Item Library",
        "♻️ Food Waste & Cost",
        "📊 Reports & Analytics",
        "📋 Attendance Records",
        "🧠 Prediction History",
        "📥 CSV Export"
    ]
)


# ============================================================
# 6. DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title(
        "🍱 AI Mess Food Demand Predictor"
    )

    st.write(
        "A Python, SQL and Machine Learning based "
        "system for predicting mess attendance "
        "and planning food quantities."
    )


    attendance = get_dataframe(
        "SELECT * FROM attendance"
    )

    predictions = get_dataframe(
        "SELECT * FROM prediction_history"
    )

    food_items = get_dataframe(
        "SELECT * FROM food_items"
    )

    waste = get_dataframe(
        "SELECT * FROM food_waste"
    )


    total_attendance_records = len(
        attendance
    )

    total_predictions = len(
        predictions
    )

    total_food_items = len(
        food_items
    )


    if waste.empty:

        total_waste_cost = 0

    else:

        total_waste_cost = (
            waste["leftover_qty"]
            * waste["cost_per_unit"]
        ).sum()


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Attendance Records",
        total_attendance_records
    )

    c2.metric(
        "Predictions",
        total_predictions
    )

    c3.metric(
        "Food Items",
        total_food_items
    )

    c4.metric(
        "Waste Cost",
        f"₹{total_waste_cost:,.2f}"
    )


    st.divider()


    st.subheader(
        "📌 How the System Works"
    )


    st.markdown("""
    1. Attendance data is stored in SQLite.
    2. The system learns from historical attendance.
    3. Random Forest Regression is used when enough data is available.
    4. The predicted diners are used by the Quantity Calculator.
    5. Food waste and cost can be tracked.
    6. Reports and CSV files help with mess management.
    """)


    st.subheader(
        "⚙️ Technology Stack"
    )


    st.info(
        "Python • Streamlit • SQLite/SQL • Pandas • "
        "NumPy • Scikit-learn • Random Forest Regression"
    )


# ============================================================
# 7. ADD ATTENDANCE
# ============================================================

elif page == "📝 Add Attendance":

    st.header(
        "📝 Add Attendance"
    )

    st.write(
        "Enter actual mess attendance so the system "
        "can learn future food demand."
    )


    with st.form("attendance_form"):

        col1, col2 = st.columns(2)


        with col1:

            meal_date = st.date_input(
                "Meal Date",
                value=date.today()
            )


            meal = st.selectbox(
                "Meal",
                [
                    "Breakfast",
                    "Lunch",
                    "Dinner"
                ]
            )


            food_type = st.selectbox(
                "Food Type",
                [
                    "Veg",
                    "Non-Veg"
                ]
            )


            menu = st.text_input(
                "Menu Name",
                placeholder="Example: Rajma Rice + Roti + Curd"
            )


            total_students = st.number_input(
                "Total Students",
                min_value=1,
                max_value=10000,
                value=300,
                step=1
            )


        with col2:

            actual_diners = st.number_input(
                "Actual Diners",
                min_value=0,
                max_value=10000,
                value=225,
                step=1
            )


            exam = st.checkbox(
                "Exam Day"
            )


            holiday = st.checkbox(
                "Holiday"
            )


            event = st.checkbox(
                "College Event"
            )


            popularity = st.slider(
                "Menu Popularity",
                min_value=1,
                max_value=5,
                value=3
            )


            notes = st.text_area(
                "Notes",
                placeholder="Optional notes"
            )


        save_attendance = st.form_submit_button(
            "💾 Save Attendance"
        )


        if save_attendance:

            if not menu.strip():

                st.error(
                    "Please enter the menu name."
                )


            elif actual_diners > total_students:

                st.error(
                    "Actual diners cannot be greater "
                    "than total students."
                )


            else:

                execute_query(
                    """
                    INSERT INTO attendance
                    (
                        meal_date,
                        meal,
                        food_type,
                        menu,
                        total_students,
                        actual_diners,
                        exam,
                        holiday,
                        event,
                        popularity,
                        notes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(meal_date),
                        meal,
                        food_type,
                        menu.strip(),
                        int(total_students),
                        int(actual_diners),
                        int(exam),
                        int(holiday),
                        int(event),
                        int(popularity),
                        notes.strip()
                    )
                )


                st.success(
                    "Attendance saved successfully."
                )


    st.divider()


    recent = get_dataframe("""
        SELECT
            id,
            meal_date,
            meal,
            food_type,
            menu,
            total_students,
            actual_diners
        FROM attendance
        ORDER BY id DESC
        LIMIT 10
    """)


    if not recent.empty:

        recent = add_display_id(
            recent
        )


        recent = recent[
            [
                "display_id",
                "meal_date",
                "meal",
                "food_type",
                "menu",
                "total_students",
                "actual_diners"
            ]
        ]


        recent.columns = [
            "ID",
            "Date",
            "Meal",
            "Food Type",
            "Menu",
            "Total Students",
            "Actual Diners"
        ]


        st.subheader(
            "Recent Attendance"
        )


        st.dataframe(
            recent,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 8. AI DEMAND PREDICTION
# ============================================================

elif page == "🤖 AI Demand Prediction":

    st.header(
        "🤖 AI Food Demand Prediction"
    )


    st.write(
        "Predict the number of students likely "
        "to eat a selected meal."
    )


    with st.form("prediction_form"):

        col1, col2 = st.columns(2)


        with col1:

            pred_meal = st.selectbox(
                "Meal",
                [
                    "Breakfast",
                    "Lunch",
                    "Dinner"
                ]
            )


            pred_food_type = st.selectbox(
                "Food Type",
                [
                    "Veg",
                    "Non-Veg"
                ]
            )


            pred_students = st.number_input(
                "Expected Students",
                min_value=1,
                max_value=10000,
                value=300,
                step=1
            )


        with col2:

            pred_exam = st.checkbox(
                "Exam Day"
            )


            pred_holiday = st.checkbox(
                "Holiday"
            )


            pred_event = st.checkbox(
                "College Event"
            )


            pred_popularity = st.slider(
                "Menu Popularity",
                min_value=1,
                max_value=5,
                value=3
            )


        predict_button = st.form_submit_button(
            "🤖 Predict Food Demand"
        )


    if predict_button:

        (
            predicted_diners,
            model_name,
            confidence
        ) = predict_diners(
            pred_meal,
            pred_food_type,
            int(pred_students),
            pred_exam,
            pred_holiday,
            pred_event,
            int(pred_popularity)
        )


        st.session_state[
            "predicted_diners"
        ] = predicted_diners


        st.session_state[
            "prediction_expected_students"
        ] = int(pred_students)


        execute_query(
            """
            INSERT INTO prediction_history
            (
                prediction_date,
                meal,
                food_type,
                expected_students,
                predicted_diners,
                model,
                confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(date.today()),
                pred_meal,
                pred_food_type,
                int(pred_students),
                predicted_diners,
                model_name,
                confidence
            )
        )


        st.success(
            "Prediction completed and saved to history."
        )


        m1, m2, m3 = st.columns(3)


        m1.metric(
            "Expected Students",
            int(pred_students)
        )


        m2.metric(
            "Predicted Diners",
            predicted_diners
        )


        m3.metric(
            "Confidence",
            confidence
        )


        st.info(
            f"Model used: {model_name}. "
            "More real attendance records generally "
            "improve reliability."
        )


        current_records = len(
            get_dataframe(
                "SELECT id FROM attendance"
            )
        )


        if current_records < 12:

            st.warning(
                "The system has fewer than 12 attendance "
                "records. This prediction is an early-data estimate."
            )


# ============================================================
# 9. MULTI-ITEM QUANTITY CALCULATOR
# ============================================================

elif page == "🍱 Quantity Calculator":

    st.header(
        "🍱 Multi-Item Menu Quantity Calculator"
    )


    st.write(
        "Calculate separate recommended quantities "
        "for every food item."
    )


    predicted_diners = st.session_state.get(
        "predicted_diners",
        None
    )


    if predicted_diners is not None:

        st.info(
            f"Latest AI prediction: "
            f"{predicted_diners} diners"
        )


    calculation_mode = st.radio(
        "Diner Count Source",
        [
            "Use Latest AI Prediction",
            "Enter Manually"
        ],
        horizontal=True
    )


    if calculation_mode == "Use Latest AI Prediction":

        if predicted_diners is None:

            st.warning(
                "No AI prediction is available yet. "
                "Please make a prediction first."
            )

            diners = 0

        else:

            diners = int(
                predicted_diners
            )


    else:

        diners = st.number_input(
            "Expected Diners",
            min_value=1,
            max_value=10000,
            value=225,
            step=1
        )


    buffer = st.selectbox(
        "Safety Buffer",
        [
            0,
            5,
            10
        ],
        format_func=lambda x: f"{x}%"
    )


    food_items = get_dataframe("""
        SELECT
            id,
            item_name,
            category,
            portion,
            unit
        FROM food_items
        ORDER BY item_name
    """)


    if food_items.empty:

        st.warning(
            "Add food items in Food Item Library first."
        )


    elif diners > 0:

        selected_names = st.multiselect(
            "Select Food Items",
            food_items["item_name"].tolist()
        )


        if selected_names:

            selected_items = food_items[
                food_items["item_name"].isin(
                    selected_names
                )
            ].copy()


            selected_items[
                "recommended_quantity"
            ] = (
                selected_items["portion"]
                * int(diners)
                * (1 + (buffer / 100))
            )


            result = selected_items[
                [
                    "item_name",
                    "category",
                    "portion",
                    "unit",
                    "recommended_quantity"
                ]
            ].copy()


            result.columns = [
                "Food Item",
                "Category",
                "Portion / Student",
                "Unit",
                "Recommended Quantity"
            ]


            st.dataframe(
                result,
                width="stretch",
                hide_index=True
            )


            st.info(
                "Formula: Portion per Student × "
                "Expected Diners × (1 + Buffer/100)"
            )


# ============================================================
# 10. FOOD ITEM LIBRARY
# ============================================================

elif page == "📚 Food Item Library":

    st.header(
        "📚 Food Item Library"
    )


    st.write(
        "Manage food items and their standard "
        "serving portions."
    )


    st.subheader(
        "➕ Add Food Item"
    )


    with st.form("food_item_form"):

        item_name = st.text_input(
            "Food Item Name",
            placeholder="Example: Rice"
        )


        category = st.selectbox(
            "Category",
            [
                "Breakfast",
                "Main Course",
                "Vegetable",
                "Dal",
                "Bread",
                "Snack",
                "Beverage",
                "Other"
            ]
        )


        portion = st.number_input(
            "Standard Portion per Student",
            min_value=0.01,
            max_value=10000.0,
            value=180.0,
            step=1.0
        )


        unit = st.selectbox(
            "Unit",
            [
                "grams",
                "kg",
                "litre",
                "ml",
                "pieces"
            ]
        )


        save_item = st.form_submit_button(
            "💾 Save Food Item"
        )


        if save_item:

            if not item_name.strip():

                st.error(
                    "Please enter the food item name."
                )


            else:

                try:

                    execute_query(
                        """
                        INSERT INTO food_items
                        (
                            item_name,
                            category,
                            portion,
                            unit
                        )
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            item_name.strip(),
                            category,
                            float(portion),
                            unit
                        )
                    )


                    st.success(
                        f"{item_name.strip()} "
                        "added successfully."
                    )


                    st.rerun()


                except sqlite3.IntegrityError:

                    st.error(
                        "This food item already exists."
                    )


    st.divider()


    st.subheader(
        "📋 Food Items"
    )


    food_items = get_dataframe("""
        SELECT
            id,
            item_name,
            category,
            portion,
            unit
        FROM food_items
        ORDER BY item_name
    """)


    if food_items.empty:

        st.info(
            "No food items added yet."
        )


    else:

        display_food_items = add_display_id(
            food_items
        )


        display_food_items = display_food_items[
            [
                "display_id",
                "item_name",
                "category",
                "portion",
                "unit"
            ]
        ]


        display_food_items.columns = [
            "ID",
            "Food Item",
            "Category",
            "Portion / Student",
            "Unit"
        ]


        st.dataframe(
            display_food_items,
            width="stretch",
            hide_index=True
        )


        st.subheader(
            "🗑️ Delete Food Item"
        )


        delete_display_id = st.selectbox(
            "Select Food Item ID",
            list(
                range(
                    1,
                    len(food_items) + 1
                )
            )
        )


        selected_food_row = food_items.iloc[
            delete_display_id - 1
        ]


        if st.button(
            "🗑️ Delete Selected Food Item"
        ):

            execute_query(
                "DELETE FROM food_items WHERE id = ?",
                (
                    int(
                        selected_food_row["id"]
                    ),
                )
            )


            st.success(
                "Food item deleted successfully."
            )


            st.rerun()


# ============================================================
# 11. FOOD WASTE & COST
# ============================================================

elif page == "♻️ Food Waste & Cost":

    st.header(
        "♻️ Food Waste & Cost Tracker"
    )


    st.write(
        "Track prepared food, leftover food "
        "and estimated waste cost."
    )


    with st.form("waste_form"):

        col1, col2 = st.columns(2)


        with col1:

            waste_date = st.date_input(
                "Date",
                value=date.today()
            )


            waste_meal = st.selectbox(
                "Meal",
                [
                    "Breakfast",
                    "Lunch",
                    "Dinner"
                ]
            )


            waste_item = st.text_input(
                "Food Item",
                placeholder="Example: Rice"
            )


            waste_unit = st.selectbox(
                "Unit",
                [
                    "kg",
                    "grams",
                    "litre",
                    "ml",
                    "pieces"
                ]
            )


        with col2:

            prepared_qty = st.number_input(
                "Prepared Quantity",
                min_value=0.01,
                max_value=100000.0,
                value=45.0,
                step=0.5
            )


            leftover_qty = st.number_input(
                "Leftover Quantity",
                min_value=0.0,
                max_value=100000.0,
                value=4.5,
                step=0.5
            )


            cost_per_unit = st.number_input(
                "Cost per Unit (₹)",
                min_value=0.0,
                max_value=100000.0,
                value=60.0,
                step=1.0
            )


        save_waste = st.form_submit_button(
            "💾 Save Waste Record"
        )


        if save_waste:

            if not waste_item.strip():

                st.error(
                    "Please enter the food item."
                )


            elif leftover_qty > prepared_qty:

                st.error(
                    "Leftover quantity cannot be "
                    "greater than prepared quantity."
                )


            else:

                execute_query(
                    """
                    INSERT INTO food_waste
                    (
                        meal_date,
                        meal,
                        item_name,
                        prepared_qty,
                        leftover_qty,
                        unit,
                        cost_per_unit
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(waste_date),
                        waste_meal,
                        waste_item.strip(),
                        float(prepared_qty),
                        float(leftover_qty),
                        waste_unit,
                        float(cost_per_unit)
                    )
                )


                st.success(
                    "Waste record saved successfully."
                )


    st.divider()


    waste_df = get_dataframe("""
        SELECT
            id,
            meal_date,
            meal,
            item_name,
            prepared_qty,
            leftover_qty,
            unit,
            cost_per_unit
        FROM food_waste
        ORDER BY id DESC
    """)


    if waste_df.empty:

        st.info(
            "No waste records available yet."
        )


    else:

        waste_df["waste_percentage"] = np.where(
            waste_df["prepared_qty"] > 0,
            (
                waste_df["leftover_qty"]
                / waste_df["prepared_qty"]
            ) * 100,
            0
        )


        waste_df["waste_cost"] = (
            waste_df["leftover_qty"]
            * waste_df["cost_per_unit"]
        )


        total_prepared = (
            waste_df["prepared_qty"].sum()
        )


        total_leftover = (
            waste_df["leftover_qty"].sum()
        )


        total_cost = (
            waste_df["waste_cost"].sum()
        )


        if total_prepared > 0:

            overall_waste_percentage = (
                total_leftover
                / total_prepared
            ) * 100

        else:

            overall_waste_percentage = 0


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Overall Waste",
            f"{overall_waste_percentage:.2f}%"
        )


        c2.metric(
            "Leftover Quantity",
            f"{total_leftover:,.2f}"
        )


        c3.metric(
            "Estimated Waste Cost",
            f"₹{total_cost:,.2f}"
        )


        st.subheader(
            "📋 Waste Records"
        )


        display_waste = waste_df.copy()


        display_waste.insert(
            0,
            "display_id",
            range(
                1,
                len(display_waste) + 1
            )
        )


        display_waste = display_waste[
            [
                "display_id",
                "meal_date",
                "meal",
                "item_name",
                "prepared_qty",
                "leftover_qty",
                "unit",
                "cost_per_unit",
                "waste_percentage",
                "waste_cost"
            ]
        ]


        display_waste.columns = [
            "ID",
            "Date",
            "Meal",
            "Food Item",
            "Prepared",
            "Leftover",
            "Unit",
            "Cost / Unit",
            "Waste %",
            "Waste Cost"
        ]


        st.dataframe(
            display_waste,
            width="stretch",
            hide_index=True
        )


        st.subheader(
            "🗑️ Delete Waste Record"
        )


        delete_waste_display_id = st.selectbox(
            "Select Waste Record ID",
            list(
                range(
                    1,
                    len(waste_df) + 1
                )
            )
        )


        selected_waste = waste_df.iloc[
            delete_waste_display_id - 1
        ]


        if st.button(
            "🗑️ Delete Selected Waste Record"
        ):

            execute_query(
                "DELETE FROM food_waste WHERE id = ?",
                (
                    int(
                        selected_waste["id"]
                    ),
                )
            )


            st.success(
                "Waste record deleted successfully."
            )


            st.rerun()


# ============================================================
# 12. REPORTS & ANALYTICS
# ============================================================

elif page == "📊 Reports & Analytics":

    st.header(
        "📊 Reports & Analytics"
    )


    attendance_df = get_dataframe("""
        SELECT
            id,
            meal_date,
            meal,
            food_type,
            total_students,
            actual_diners,
            exam,
            holiday,
            event,
            popularity
        FROM attendance
        ORDER BY meal_date
    """)


    if attendance_df.empty:

        st.info(
            "Add attendance records to generate reports."
        )


    else:

        total_students = (
            attendance_df[
                "total_students"
            ].sum()
        )


        actual_diners = (
            attendance_df[
                "actual_diners"
            ].sum()
        )


        if total_students > 0:

            avg_attendance = (
                actual_diners
                / total_students
            ) * 100

        else:

            avg_attendance = 0


        average_diners = (
            attendance_df[
                "actual_diners"
            ].mean()
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Attendance Records",
            len(attendance_df)
        )


        c2.metric(
            "Average Attendance",
            f"{avg_attendance:.2f}%"
        )


        c3.metric(
            "Average Diners",
            f"{average_diners:.1f}"
        )


        st.divider()


        # ----------------------------------------------------
        # GRAPH 1
        # ----------------------------------------------------

        st.subheader(
            "📈 Graph 1 — Attendance Trend"
        )


        trend = (
            attendance_df
            .groupby(
                "meal_date"
            )[
                "actual_diners"
            ]
            .sum()
        )


        st.line_chart(
            trend
        )


        # ----------------------------------------------------
        # GRAPH 2
        # ----------------------------------------------------

        st.subheader(
            "📊 Graph 2 — Meal-wise Attendance"
        )


        meal_summary = (
            attendance_df
            .groupby(
                "meal"
            )[
                "actual_diners"
            ]
            .mean()
        )


        st.bar_chart(
            meal_summary
        )


        # ----------------------------------------------------
        # EXAM VS NORMAL
        # ----------------------------------------------------

        st.subheader(
            "📝 Exam vs Normal Attendance"
        )


        exam_summary = (
            attendance_df.copy()
        )


        exam_summary["Day Type"] = np.where(
            exam_summary["exam"] == 1,
            "Exam Day",
            "Normal Day"
        )


        exam_summary = (
            exam_summary
            .groupby(
                "Day Type"
            )[
                "actual_diners"
            ]
            .mean()
            .reset_index()
        )


        exam_summary.columns = [
            "Day Type",
            "Average Diners"
        ]


        st.dataframe(
            exam_summary,
            width="stretch",
            hide_index=True
        )


        # ----------------------------------------------------
        # DETAILED REPORT
        # ----------------------------------------------------

        st.subheader(
            "📋 Detailed Attendance Report"
        )


        detailed = add_display_id(
            attendance_df
        )


        detailed = detailed[
            [
                "display_id",
                "meal_date",
                "meal",
                "food_type",
                "total_students",
                "actual_diners",
                "exam",
                "holiday",
                "event",
                "popularity"
            ]
        ]


        detailed.columns = [
            "ID",
            "Date",
            "Meal",
            "Food Type",
            "Total Students",
            "Actual Diners",
            "Exam",
            "Holiday",
            "Event",
            "Popularity"
        ]


        st.dataframe(
            detailed,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 13. ATTENDANCE RECORDS
# ============================================================

elif page == "📋 Attendance Records":

    st.header(
        "📋 Attendance Records"
    )


    st.write(
        "View, search and delete individual "
        "attendance records."
    )


    attendance_records = get_dataframe("""
        SELECT
            id,
            meal_date,
            meal,
            food_type,
            menu,
            total_students,
            actual_diners,
            exam,
            holiday,
            event,
            popularity,
            notes
        FROM attendance
        ORDER BY meal_date DESC, id DESC
    """)


    if attendance_records.empty:

        st.info(
            "No attendance records available yet."
        )


    else:

        search_text = st.text_input(
            "🔎 Search by Meal, Food Type or Menu",
            placeholder="Example: Lunch, Veg, Rice"
        )


        filtered_records = (
            attendance_records.copy()
        )


        if search_text.strip():

            search_text = (
                search_text
                .strip()
                .lower()
            )


            filtered_records = (
                filtered_records[
                    filtered_records[
                        [
                            "meal",
                            "food_type",
                            "menu"
                        ]
                    ]
                    .astype(str)
                    .apply(
                        lambda row:
                        row.str.lower()
                        .str.contains(
                            search_text,
                            regex=False
                        )
                        .any(),
                        axis=1
                    )
                ]
            )


        st.subheader(
            "💾 Saved Attendance"
        )


        if not filtered_records.empty:

            display_records = (
                filtered_records.copy()
            )


            display_records.insert(
                0,
                "display_id",
                range(
                    1,
                    len(display_records) + 1
                )
            )


            display_records = display_records[
                [
                    "display_id",
                    "meal_date",
                    "meal",
                    "food_type",
                    "menu",
                    "total_students",
                    "actual_diners",
                    "exam",
                    "holiday",
                    "event",
                    "popularity",
                    "notes"
                ]
            ]


            display_records.columns = [
                "ID",
                "Date",
                "Meal",
                "Food Type",
                "Menu",
                "Total Students",
                "Actual Diners",
                "Exam",
                "Holiday",
                "Event",
                "Popularity",
                "Notes"
            ]


            st.dataframe(
                display_records,
                width="stretch",
                hide_index=True
            )


            st.divider()


            st.subheader(
                "🗑️ Delete Attendance Record"
            )


            delete_display_id = st.selectbox(
                "Select Attendance Record ID",
                list(
                    range(
                        1,
                        len(filtered_records) + 1
                    )
                )
            )


            confirm_delete = st.checkbox(
                "I confirm that I want to delete this record."
            )


            if st.button(
                "🗑️ Delete Selected Record"
            ):

                if confirm_delete:

                    # Actual SQLite ID
                    real_id = int(
                        filtered_records.iloc[
                            delete_display_id - 1
                        ]["id"]
                    )


                    execute_query(
                        "DELETE FROM attendance WHERE id = ?",
                        (real_id,)
                    )


                    st.success(
                        "Attendance record deleted successfully."
                    )


                    st.rerun()


                else:

                    st.warning(
                        "Please confirm deletion first."
                    )


        else:

            st.info(
                "No matching records found."
            )


# ============================================================
# 14. PREDICTION HISTORY
# ============================================================

elif page == "🧠 Prediction History":

    st.header(
        "🧠 Prediction History"
    )


    st.write(
        "View the history of AI food demand predictions."
    )


    prediction_df = get_dataframe("""
        SELECT
            id,
            prediction_date,
            meal,
            food_type,
            expected_students,
            predicted_diners,
            model,
            confidence
        FROM prediction_history
        ORDER BY id DESC
    """)


    if prediction_df.empty:

        st.info(
            "No prediction history available yet."
        )


    else:

        total_predictions = len(
            prediction_df
        )


        average_predicted = (
            prediction_df[
                "predicted_diners"
            ].mean()
        )


        latest_prediction = (
            prediction_df.iloc[0][
                "predicted_diners"
            ]
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Total Predictions",
            total_predictions
        )


        c2.metric(
            "Average Predicted Diners",
            f"{average_predicted:.1f}"
        )


        c3.metric(
            "Latest Prediction",
            int(latest_prediction)
        )


        st.divider()


        display_predictions = (
            add_display_id(
                prediction_df
            )
        )


        display_predictions = (
            display_predictions[
                [
                    "display_id",
                    "prediction_date",
                    "meal",
                    "food_type",
                    "expected_students",
                    "predicted_diners",
                    "model",
                    "confidence"
                ]
            ]
        )


        display_predictions.columns = [
            "ID",
            "Date",
            "Meal",
            "Food Type",
            "Expected Students",
            "Predicted Diners",
            "Model",
            "Confidence"
        ]


        st.dataframe(
            display_predictions,
            width="stretch",
            hide_index=True
        )


        st.info(
            "Prediction History is view-only. "
            "Historical predictions are preserved for analysis."
        )


# ============================================================
# 15. CSV EXPORT
# ============================================================

elif page == "📥 CSV Export":

    st.header(
        "📥 CSV Export"
    )


    st.write(
        "Download project data as CSV files."
    )


    attendance_export = get_dataframe("""
        SELECT
            meal_date,
            meal,
            food_type,
            menu,
            total_students,
            actual_diners,
            exam,
            holiday,
            event,
            popularity,
            notes
        FROM attendance
        ORDER BY meal_date DESC
    """)


    prediction_export = get_dataframe("""
        SELECT
            prediction_date,
            meal,
            food_type,
            expected_students,
            predicted_diners,
            model,
            confidence
        FROM prediction_history
        ORDER BY id DESC
    """)


    waste_export = get_dataframe("""
        SELECT
            meal_date,
            meal,
            item_name,
            prepared_qty,
            leftover_qty,
            unit,
            cost_per_unit
        FROM food_waste
        ORDER BY id DESC
    """)


    st.subheader(
        "📝 Attendance Data"
    )


    st.download_button(
        "⬇️ Download Attendance CSV",
        data=attendance_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="attendance_records.csv",
        mime="text/csv"
    )


    st.subheader(
        "🧠 Prediction History"
    )


    st.download_button(
        "⬇️ Download Prediction History CSV",
        data=prediction_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv"
    )


    st.subheader(
        "♻️ Food Waste Data"
    )


    st.download_button(
        "⬇️ Download Food Waste CSV",
        data=waste_export.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="food_waste_records.csv",
        mime="text/csv"
    )


    st.success(
        "CSV export is ready. "
        "The downloaded files can be opened in Excel."
    )
