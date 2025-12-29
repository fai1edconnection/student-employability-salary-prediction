# 🎓 Student Employability & Salary Prediction Using Machine Learning

A comprehensive machine learning project that predicts student placement outcomes and salary ranges based on academic performance, work experience, and other key factors. This research provides data-driven insights for students, educational institutions, and employers in the Saudi Arabian context.

## 📋 Project Overview

This project develops predictive models to answer two critical questions students face:
1. **Will I get placed after graduation?** (Classification task)
2. **What salary should I expect?** (Regression task)

By integrating multiple datasets and applying various machine learning algorithms, we provide actionable insights that can enhance career counseling and academic planning.

## 🎯 Key Features

- **Dual Prediction Framework**: Classification for placement (Placed/Not Placed) and regression for salary prediction
- **Multi-Source Data Integration**: Combines campus placement data, industry salary benchmarks, and global developer survey data
- **Comprehensive Model Evaluation**: Tests 10+ ML algorithms including Logistic Regression, Decision Trees, Random Forest, SVM, Gradient Boosting, and more
- **Feature Importance Analysis**: Identifies key factors affecting employability and salary
- **Saudi Arabian Context**: Data normalized to SAR and tailored to local market conditions
- **Interactive Interface**: Web-based tool for live predictions (see project slides)

## 📊 Results Summary

### Classification (Placement Prediction)
- **Best Model**: Random Forest Classifier
- **Accuracy**: 83.72%
- **Precision/Recall/F1**: 87.50%

### Regression (Salary Prediction)  
- **Best Model**: Random Forest Regressor
- **RMSE**: 3,234 SAR (7.5% relative error)
- **MAE**: 2,418 SAR
- **89.2%** of predictions within ±5,000 SAR of actual values

### Key Insights
- **Academic Performance**: 45.23% importance for placement prediction
- **Work Experience**: Increases placement probability by 31 percentage points
- **Experience Years**: 62.34% importance for salary determination
- **Gender**: Minimal influence on outcomes (equitable predictions)

## 📁 Repository Structure
```
├── notebook/              # Jupyter notebook for EDA and modeling
├── models/                 # Trained model files
├── docs/                  # Research paper & Project slides and demo 
└── README.md              # This file
```

## 🛠️ Technologies Used

- **Programming**: Python 3.8+
- **Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Data Processing**: Feature engineering, normalization, cross-validation
- **Evaluation**: Accuracy, Precision, Recall, F1, RMSE, MAE, R²

## 📈 Methodology

1. **Data Collection**: Integrated three datasets (Campus Placement, Data Science Salaries, Stack Overflow Survey)
2. **Preprocessing**: Currency normalization to SAR, missing value handling, outlier removal
3. **Feature Engineering**: Created Academic Score (weighted average), encoded categorical variables
4. **Model Training**: 80/20 train-test split with cross-validation
5. **Evaluation**: Comprehensive metrics for both classification and regression tasks
6. **Interpretation**: Feature importance analysis and practical implications

## 💡 Practical Applications

### For Students:
- Prioritize academic excellence and seek internships
- Set realistic salary expectations (22,000-64,000 SAR range)
- Understand which factors most impact employability

### For Institutions:
- Integrate mandatory internships into curricula
- Implement data-driven career counseling
- Track placement outcomes for continuous improvement

### For Employers:
- Identify high-potential candidates using predictive models
- Structure compensation based on market data

## 📚 Research Contributions

- First comprehensive employability study for the Saudi market
- Systematic evaluation of 10 ML algorithms across dual prediction tasks
- Integration of academic, industry, and global perspectives
- Ready-to-deploy predictive framework with practical utility

## 📄 Documentation

- **Research Paper**: Complete methodology, results, and analysis in `Data_FinalPaper2.pdf`
- **Presentation Slides**: Project overview and key findings in `DataSci_FinalProject_Slides.pdf`
- **Code Documentation**: In-line comments and notebook explanations

## 👥 Contributors

- Sarah Eid | Judy Abuquata | Nancy Elhaddad | Passent Elkafrawy
- Effat University, College of Engineering, Computer Science Department
- Jeddah, Saudi Arabia

## 📄 License

This project is for academic research purposes. Please cite appropriately if used.

## 🔗 References

- Campus Recruitment Dataset: [Kaggle](https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement)
- Data Science Job Salaries: [Kaggle](https://www.kaggle.com/datasets/nich1788/data-science-job-salaries)
- Stack Overflow Developer Survey: [2024 Insights](https://survey.stackoverflow.co/)

---

**Tags**: `machine-learning` `education` `career-analytics` `salary-prediction` `employability` `data-science` `saudi-arabia` `student-success` `random-forest` `classification` `regression`
