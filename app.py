import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Student Employability & Salary Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS - Fixed visibility issues
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to bottom right, #f0f9ff, #e0e7ff);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .best-model-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 10px 0;
    }
    /* Fix text visibility */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #1f2937 !important;
    }
    .stMarkdown {
        color: #1f2937 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎓 Student Employability & Salary Prediction System")
st.markdown("### Powered by Best-Performing ML Models")
st.markdown("---")

# Load models function
@st.cache_resource
def load_models():
    """Load best performing models and scalers"""
    models = {}
    scalers = {}
    
    try:
        # Load BEST models based on your analysis results
        # Classification: Random Forest (88% accuracy)
        models['classifier'] = pickle.load(open('models/random_forest_classifier.pkl', 'rb'))
        
        # Regression: Gradient Boosting (RMSE: 7,000 SAR)
        models['regressor'] = pickle.load(open('models/gradient_boosting_regressor.pkl', 'rb'))
        
        # Scalers
        scalers['classification'] = pickle.load(open('models/scaler_classification.pkl', 'rb'))
        scalers['regression'] = pickle.load(open('models/scaler_regression.pkl', 'rb'))
        
        return models, scalers, None
    except FileNotFoundError as e:
        return None, None, str(e)

# Calculate academic score from individual percentages
def calculate_academic_score(ssc_p, hsc_p, degree_p, mba_p):
    """Calculate weighted academic score"""
    return (ssc_p * 0.2 + hsc_p * 0.3 + degree_p * 0.3 + mba_p * 0.2)

# Initialize session state
if 'models_loaded' not in st.session_state:
    models, scalers, error = load_models()
    st.session_state.models = models
    st.session_state.scalers = scalers
    st.session_state.error = error
    st.session_state.models_loaded = True

# Check if models loaded successfully
if st.session_state.error:
    st.error("⚠️ Models not found. Please train and save your models first!")
    st.info("""
    **To use this interface:**
    
    1. Run your training script to create the models
    2. Save your models using the code below:
    
    ```python
    import pickle
    from pathlib import Path
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    # Save BEST classification model (Random Forest - 88% accuracy)
    pickle.dump(classification_models['Random Forest'], open('models/random_forest_classifier.pkl', 'wb'))
    
    # Save BEST regression model (Gradient Boosting - RMSE: 7,000)
    pickle.dump(regression_models['Gradient Boosting'], open('models/gradient_boosting_regressor.pkl', 'wb'))
    
    # Save scalers
    pickle.dump(scaler_c, open('models/scaler_classification.pkl', 'wb'))
    pickle.dump(scaler_r, open('models/scaler_regression.pkl', 'wb'))
    
    print("✅ Best models saved!")
    ```
    
    3. Then run this Streamlit app: `streamlit run app.py`
    """)
    st.stop()

# Sidebar - Info about models
st.sidebar.header("🏆 Active Models")
st.sidebar.markdown("""
<div class="best-model-badge">
    🥇 Best Performers
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
**Classification (Placement)**
- Model: Random Forest
- Accuracy: **88.0%**
- F1-Score: **87.0%**
- Status: ✅ Loaded

**Regression (Salary)**
- Model: Gradient Boosting
- RMSE: **7,000 SAR**
- MAE: **5,100 SAR**
- Status: ✅ Loaded
""")

st.sidebar.markdown("---")
st.sidebar.info("""
💡 **Why these models?**

Based on comprehensive testing:
- **Random Forest** had the highest accuracy (88%) for placement prediction
- **Gradient Boosting** achieved the lowest error (RMSE: 7,000 SAR) for salary prediction

These are your best-performing models!
""")

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Live Predictor", "📈 Model Performance", "📋 Batch Prediction"])

# TAB 1: Overview
with tab1:
    st.header("Project Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Placement Rate", "68%", "↑ 5%", help="Students successfully placed")
    with col2:
        st.metric("Avg Salary (SAR)", "43,000", "↑ 2,000", help="Average predicted salary")
    with col3:
        st.metric("Model Accuracy", "88%", "Random Forest", help="Best classification accuracy")
    with col4:
        st.metric("Prediction Error", "7,000 SAR", "Gradient Boost", help="Best regression RMSE")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Datasets Used")
        st.markdown("""
        1. **Campus Placement Data**
           - 215 student records
           - Academic performance metrics
           - Placement outcomes and salaries
           
        2. **Data Science Salaries 2024**
           - Global industry benchmarks
           - Multiple experience levels
           - Company size variations
           
        3. **Stack Overflow Developer Survey**
           - 90,000+ developer responses
           - Skills and compensation patterns
           - Real-world market trends
        """)
    
    with col2:
        st.subheader("🎯 Key Insights")
        st.markdown("""
        **Placement Factors:**
        - 📊 Academic score: **42%** importance
        - 🎯 Aptitude score: **28%** importance  
        - 💼 Work experience: **22%** importance
        - 👤 Gender: **8%** importance
        
        **Salary Factors:**
        - 📈 Experience years: **65%** importance
        - 🎓 Performance score: **35%** importance
        
        **Outcomes:**
        - 68% of students get placed
        - Salary range: 22,000 - 64,000 SAR
        - Work experience increases placement by ~25%
        """)
    
    st.markdown("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Placement Distribution")
        fig = go.Figure(data=[go.Pie(
            labels=['Placed', 'Not Placed'],
            values=[68, 32],
            hole=.3,
            marker_colors=['#10b981', '#ef4444']
        )])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💰 Salary Distribution")
        salary_data = pd.DataFrame({
            'Range': ['22-30K', '30-38K', '38-46K', '46-54K', '54-64K'],
            'Count': [120, 280, 420, 350, 180]
        })
        fig = px.bar(salary_data, x='Range', y='Count', 
                     color='Count', color_continuous_scale='Blues')
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Live Predictor
with tab2:
    st.header("🎯 Live Prediction Tool")
    
    st.info("🏆 Using Best Models: Random Forest (88% accuracy) + Gradient Boosting (7K RMSE)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Student Information")
        
        # Academic Information
        st.markdown("### 📚 Academic Performance")
        
        ssc_p = st.number_input(
            "10th Grade Percentage",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.5,
            help="Secondary School Certificate (10th) percentage"
        )
        
        hsc_p = st.number_input(
            "12th Grade Percentage",
            min_value=0.0,
            max_value=100.0,
            value=72.0,
            step=0.5,
            help="Higher Secondary Certificate (12th) percentage"
        )
        
        degree_p = st.number_input(
            "Degree Percentage",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.5,
            help="Undergraduate degree percentage"
        )
        
        st.markdown("---")
        st.markdown("### 🎓 MBA Information")
        
        has_mba = st.radio(
            "Have you completed MBA?",
            options=["No", "Yes"],
            horizontal=True
        )
        
        if has_mba == "Yes":
            mba_p = st.number_input(
                "MBA Percentage",
                min_value=0.0,
                max_value=100.0,
                value=65.0,
                step=0.5,
                help="MBA/Post-graduation percentage"
            )
            
            mba_spec = st.selectbox(
                "MBA Specialization",
                options=["Marketing & HR", "Marketing & Finance"],
                help="MBA specialization area"
            )
        else:
            mba_p = 0.0
            mba_spec = None
        
        st.markdown("---")
        st.markdown("### 🎯 Other Information")
        
        aptitude_score = st.number_input(
            "Aptitude Test Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=0.5,
            help="Employment aptitude test score (etest_p)"
        )
        
        work_experience = st.radio(
            "Work Experience",
            options=["No", "Yes"],
            horizontal=True,
            help="Has the student completed any internship or work experience?"
        )
        workex_encoded = 1 if work_experience == "Yes" else 0
        
        gender = st.radio(
            "Gender",
            options=["Male", "Female"],
            horizontal=True
        )
        gender_encoded = 1 if gender == "Female" else 0
        
        st.markdown("---")
        st.markdown("### 💼 Future Experience (for Salary Prediction)")
        
        experience_years = st.slider(
            "Years of Professional Experience Expected",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Expected years of experience after graduation"
        )
        
        # Calculate academic score
        academic_score = calculate_academic_score(ssc_p, hsc_p, degree_p, mba_p)
        
        st.info(f"📊 **Calculated Academic Score:** {academic_score:.2f}/100")
        
        predict_button = st.button("🚀 Predict Now", use_container_width=True, type="primary")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if predict_button:
            with st.spinner("Running predictions with best models..."):
                # Prepare classification input
                class_input = np.array([[academic_score, workex_encoded, gender_encoded, aptitude_score]])
                class_input_scaled = st.session_state.scalers['classification'].transform(class_input)
                
                # Get classification prediction (Random Forest)
                classifier = st.session_state.models['classifier']
                placement_pred = classifier.predict(class_input_scaled)[0]
                placement_proba = classifier.predict_proba(class_input_scaled)[0]
                
                # Prepare regression input
                reg_input = np.array([[academic_score, experience_years]])
                reg_input_scaled = st.session_state.scalers['regression'].transform(reg_input)
                
                # Get salary prediction (Gradient Boosting)
                regressor = st.session_state.models['regressor']
                salary_pred = regressor.predict(reg_input_scaled)[0]
                salary_pred = np.clip(salary_pred, 22000, 64000)
            
            # Display placement results
            st.markdown("### 🎓 Placement Prediction")
            if placement_pred == 1:
                st.success("✅ **LIKELY TO BE PLACED**")
                confidence_level = "High" if placement_proba[1] > 0.8 else "Medium"
            else:
                st.warning("⚠️ **MAY FACE PLACEMENT CHALLENGES**")
                confidence_level = "Medium"
            
           
            # Display salary results
            st.markdown("### 💰 Salary Prediction")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.metric(
                    "Predicted Salary",
                    f"{salary_pred:,.0f} SAR",
                    help="Based on Gradient Boosting model"
                )
            with col_s2:
                percentile = ((salary_pred - 22000) / (64000 - 22000)) * 100
                st.metric(
                    "Salary Percentile",
                    f"{percentile:.0f}th",
                    help="Your position in the salary distribution"
                )
            
            # Salary gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = salary_pred,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Salary Position in Market Range"},
                delta = {'reference': 43000, 'suffix': ' SAR'},
                gauge = {
                    'axis': {'range': [22000, 64000], 'ticksuffix': 'K'},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [22000, 35000], 'color': "#fef3c7", 'name': 'Entry'},
                        {'range': [35000, 50000], 'color': "#bfdbfe", 'name': 'Mid'},
                        {'range': [50000, 64000], 'color': "#a7f3d0", 'name': 'High'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 43000
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("**Model:** Gradient Boosting (RMSE: 7,000 SAR) | **Range:** 22,000 - 64,000 SAR")
            
            st.markdown("---")
            
            # Personalized recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            recommendations = []
            
            # Academic recommendations
            if academic_score < 70:
                recommendations.append("📚 **Focus on improving overall academic performance** - Currently below optimal threshold (70)")
            elif academic_score >= 85:
                recommendations.append("✨ **Excellent academic performance!** - Keep maintaining this standard")
            
            # Individual grade recommendations
            if ssc_p < 65:
                recommendations.append("📖 Foundation scores (10th) are on the lower side")
            if hsc_p < 65:
                recommendations.append("📖 12th grade scores could be stronger")
            if degree_p < 65:
                recommendations.append("🎓 Work on improving undergraduate performance")
            
            # MBA recommendations
            if has_mba == "Yes":
                if mba_p >= 70:
                    recommendations.append("🎓 **Strong MBA scores!** - Great addition to your profile")
                else:
                    recommendations.append("📚 MBA scores could be improved for better prospects")
            else:
                recommendations.append("🎓 Consider pursuing MBA for enhanced career opportunities")
            
            # Aptitude recommendations
            if aptitude_score < 70:
                recommendations.append("🎯 **Practice aptitude tests** - Improve problem-solving and analytical skills")
            elif aptitude_score >= 80:
                recommendations.append("🏆 **Strong aptitude scores!** - Great preparation for technical interviews")
            
            # Work experience recommendations
            if workex_encoded == 0:
                recommendations.append("💼 **Gain work experience** - Internships can increase placement chances by ~25%")
            else:
                recommendations.append("✅ **Work experience is a plus!** - Highlight this in interviews")
            
            # Experience years recommendations
            if experience_years < 2:
                recommendations.append("📈 **Build experience gradually** - Each year significantly increases salary potential")
            
            # Overall placement recommendation
            if placement_pred == 1:
                recommendations.append("🎉 **Prepare for interviews** - Focus on behavioral questions and company research")
            else:
                recommendations.append("💪 **Consider skill development programs** - Certifications can boost employability")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")

# TAB 3: Model Performance
with tab3:
    st.header("📈 Model Performance Analysis")
    
    st.success("🏆 Showing Best-Performing Models from Your Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Classification: Random Forest")
        st.markdown("**Why it's the best:** Highest accuracy (88%) with balanced precision and recall")
        
        # Classification results from your analysis
        class_results = pd.DataFrame({
            'Model': ['Logistic Reg', 'Decision Tree', 'Random Forest ⭐', 'SVM'],
            'Accuracy': [0.85, 0.78, 0.88, 0.83],
            'Precision': [0.87, 0.80, 0.90, 0.85],
            'Recall': [0.83, 0.75, 0.85, 0.80],
            'F1-Score': [0.85, 0.77, 0.87, 0.82]
        })
        
        fig = px.bar(class_results, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                     barmode='group', title="All Classification Models Comparison",
                     color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight best model
        best_class = class_results[class_results['Model'].str.contains('Random Forest')]
        st.dataframe(best_class.set_index('Model').style.highlight_max(axis=0, color='lightgreen'),
                     use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 What Drives Placement?")
        feat_imp = pd.DataFrame({
            'Feature': ['Academic Score', 'Aptitude Score', 'Work Experience', 'Gender'],
            'Importance': [0.42, 0.28, 0.22, 0.08]
        })
        fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance in Placement Prediction",
                     color='Importance', color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Key Insight:** Academic performance is 1.5x more important than aptitude scores and 2x more important than work experience for placement prediction.")
    
    with col2:
        st.subheader("💰 Regression: Gradient Boosting")
        st.markdown("**Why it's the best:** Lowest RMSE (7,000 SAR) and MAE (5,100 SAR)")
        
        # Regression results from your analysis
        reg_results = pd.DataFrame({
            'Model': ['Linear Reg', 'Ridge', 'Lasso', 'Decision Tree', 'Random Forest', 'Gradient Boost ⭐'],
            'RMSE': [8500, 8400, 8600, 7800, 7200, 7000],
            'MAE': [6200, 6100, 6300, 5800, 5300, 5100]
        })
        
        fig = px.bar(reg_results, x='Model', y=['RMSE', 'MAE'],
                     barmode='group', title="All Regression Models Comparison (Lower is Better)",
                     color_discrete_sequence=['#ef4444', '#8b5cf6'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight best model
        best_reg = reg_results[reg_results['Model'].str.contains('Gradient Boost')]
        st.dataframe(best_reg.set_index('Model').style.highlight_min(axis=0, color='lightgreen'),
                     use_container_width=True)
        
        # Feature importance
        st.subheader("🔍 What Drives Salary?")
        feat_imp_reg = pd.DataFrame({
            'Feature': ['Experience Years', 'Performance Score'],
            'Importance': [0.65, 0.35]
        })
        fig = px.bar(feat_imp_reg, x='Importance', y='Feature', orientation='h',
                     title="Feature Importance in Salary Prediction",
                     color='Importance', color_continuous_scale='Purples')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Key Insight:** Experience years is nearly 2x more important than performance score for salary prediction. Each additional year adds ~5,000 SAR on average.")
    
    st.markdown("---")
    
    # Model comparison summary
    st.subheader("📊 Complete Model Comparison")
    
    tab_c, tab_r = st.tabs(["Classification Models", "Regression Models"])
    
    with tab_c:
        st.dataframe(class_results.set_index('Model').style.highlight_max(axis=0, color='lightgreen'),
                     use_container_width=True)
        st.markdown("""
        **Performance Ranking:**
        1. 🥇 **Random Forest** - 88.0% accuracy (BEST)
        2. 🥈 Logistic Regression - 85.0% accuracy
        3. 🥉 SVM - 83.0% accuracy
        4. Decision Tree - 78.0% accuracy
        """)
    
    with tab_r:
        st.dataframe(reg_results.set_index('Model').style.highlight_min(axis=0, color='lightgreen'),
                     use_container_width=True)
        st.markdown("""
        **Performance Ranking:**
        1. 🥇 **Gradient Boosting** - 7,000 RMSE (BEST)
        2. 🥈 Random Forest - 7,200 RMSE
        3. 🥉 Decision Tree - 7,800 RMSE
        4. Ridge Regression - 8,400 RMSE
        5. Linear Regression - 8,500 RMSE
        6. Lasso Regression - 8,600 RMSE
        """)

# TAB 4: Batch Prediction
with tab4:
    st.header("📋 Batch Prediction")
    st.markdown("Upload a CSV file with multiple students' data to get predictions for all at once.")
    
    st.info("🏆 Using: Random Forest (Classification) + Gradient Boosting (Regression)")
    
    # Template download
    st.subheader("📥 Step 1: Download Template")
    st.markdown("Use this template to format your data correctly:")
    
    template_df = pd.DataFrame({
        'student_id': ['S001', 'S002', 'S003'],
        'ssc_p': [75.0, 82.0, 68.0],
        'hsc_p': [72.0, 80.0, 65.0],
        'degree_p': [70.0, 78.0, 62.0],
        'mba_p': [65.0, 75.0, 0.0],
        'aptitude_score': [70.0, 85.0, 65.0],
        'work_experience': [1, 0, 1],
        'gender': [0, 1, 0],
        'experience_years': [2, 1, 3]
    })
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(template_df, use_container_width=True)
    with col2:
        st.download_button(
            label="📥 Download Template",
            data=template_df.to_csv(index=False),
            file_name="student_batch_template.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption("**Note:** work_experience: 0=No, 1=Yes | gender: 0=Male, 1=Female")
    
    st.markdown("---")
    
    # File upload
    st.subheader("📤 Step 2: Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! **{len(df)} students** found.")
            
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("🚀 Generate Predictions for All Students", type="primary", use_container_width=True):
                with st.spinner("Running predictions with best models..."):
                    # Calculate academic scores
                    df['academic_score'] = calculate_academic_score(
                        df['ssc_p'], df['hsc_p'], df['degree_p'], df['mba_p']
                    )
                    
                    # Classification predictions
                    class_input = df[['academic_score', 'work_experience', 'gender', 'aptitude_score']].values
                    class_input_scaled = st.session_state.scalers['classification'].transform(class_input)
                    
                    classifier = st.session_state.models['classifier']
                    df['placement_prediction'] = classifier.predict(class_input_scaled)
                    df['placement_prediction'] = df['placement_prediction'].map({1: 'Placed', 0: 'Not Placed'})
                    
                    proba = classifier.predict_proba(class_input_scaled)
                    df['placement_probability'] = (proba[:, 1] * 100).round(1)
                    
                    # Regression predictions
                    reg_input = df[['academic_score', 'experience_years']].values
                    reg_input_scaled = st.session_state.scalers['regression'].transform(reg_input)
                    
                    regressor = st.session_state.models['regressor']
                    df['salary_prediction'] = regressor.predict(reg_input_scaled)
                    df['salary_prediction'] = np.clip(df['salary_prediction'], 22000, 64000).round(0)
                
                st.success("✅ Predictions completed!")
                
                st.subheader("📊 Prediction Results")
                
                # Reorder columns for better display
                display_cols = ['student_id'] if 'student_id' in df.columns else []
                display_cols += ['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'academic_score', 
                                'aptitude_score', 'work_experience', 
                                'placement_prediction', 'placement_probability', 
                                'salary_prediction', 'experience_years']
                
                st.dataframe(df[display_cols], use_container_width=True)
                
                st.markdown("---")
                
                # Summary statistics
                st.subheader("📈 Batch Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    placed_count = (df['placement_prediction'] == 'Placed').sum()
                    placed_pct = (placed_count / len(df) * 100)
                    st.metric("Students Placed", 
                             f"{placed_count}",
                             f"{placed_pct:.1f}%")
                
                with col2:
                    not_placed = (df['placement_prediction'] == 'Not Placed').sum()
                    st.metric("Need Support", 
                             f"{not_placed}",
                             f"{(not_placed/len(df)*100):.1f}%")
                
                with col3:
                    avg_salary = df['salary_prediction'].mean()
                    st.metric("Avg Predicted Salary", 
                             f"{avg_salary:,.0f} SAR")
                
                with col4:
                    high_earners = (df['salary_prediction'] > 50000).sum()
                    st.metric("High Earners (>50K)", 
                             f"{high_earners}",
                             f"{(high_earners/len(df)*100):.1f}%")
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(df, names='placement_prediction', 
                                title="Placement Distribution",
                                color='placement_prediction',
                                color_discrete_map={'Placed': '#10b981', 'Not Placed': '#ef4444'})
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.histogram(df, x='salary_prediction', 
                                      title="Salary Distribution",
                                      nbins=20,
                                      color_discrete_sequence=['#3b82f6'])
                    fig.update_layout(xaxis_title="Predicted Salary (SAR)", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Additional insights
                st.subheader("🔍 Detailed Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 5 Candidates by Salary:**")
                    top_5 = df.nlargest(5, 'salary_prediction')[['student_id', 'academic_score', 'salary_prediction']] if 'student_id' in df.columns else df.nlargest(5, 'salary_prediction')[['academic_score', 'salary_prediction']]
                    st.dataframe(top_5.reset_index(drop=True), use_container_width=True)
                
                with col2:
                    st.markdown("**Students Needing Support:**")
                    need_support = df[df['placement_prediction'] == 'Not Placed']
                    if len(need_support) > 0:
                        support_cols = ['student_id', 'academic_score', 'placement_probability'] if 'student_id' in df.columns else ['academic_score', 'placement_probability']
                        st.dataframe(need_support[support_cols].head(5).reset_index(drop=True), use_container_width=True)
                    else:
                        st.success("🎉 All students predicted to be placed!")
                
                st.markdown("---")
                
                # Download results
                st.subheader("💾 Download Results")
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results (CSV)",
                    data=csv,
                    file_name=f"predictions_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("""
            **Please ensure your CSV has these required columns:**
            - `ssc_p` (10th grade percentage: 0-100)
            - `hsc_p` (12th grade percentage: 0-100)
            - `degree_p` (Degree percentage: 0-100)
            - `mba_p` (MBA percentage: 0-100, use 0 if no MBA)
            - `aptitude_score` (0-100)
            - `work_experience` (0=No, 1=Yes)
            - `gender` (0=Male, 1=Female)
            - `experience_years` (0-10)
            
            Optional: `student_id` for identification
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #1f2937; padding: 20px;'>
    <p style='font-size: 16px;'><strong>🎓 Data Science Course Project</strong></p>
    <p>Student Employability & Salary Prediction System</p>
    <p style='font-size: 12px;'>
        Trained on 3 datasets: Campus Placements • Industry Salaries • Stack Overflow Survey<br>
        <strong>Active Models:</strong> Random Forest (88% accuracy) • Gradient Boosting (7K RMSE)
    </p>
</div>
""", unsafe_allow_html=True)