# Best_BG_performance
Understanding how blood glucose (BG) extremes influence the **Heart‑Rate–Power (HR–P)** relationship in endurance cycling.

This project analyzes cycling performance data together with Nightscout CGM traces to quantify how metabolic state affects aerobic efficiency. The repository includes Jupyter notebooks, utilities, and example workflows for correlating BG levels with HR–P dynamics.

---

## 📌 Project Goals

- Quantify how **hyperglycemia** and **hypoglycemia** shift the HR–P curve  
- Correlate **Nightscout CGM data** with cycling activity files  
- Analyze **interval performance** under different metabolic states  
- Visualize HR–P efficiency changes across BG ranges  
- Provide a reproducible workflow for athletes 

---

## 📁 Repository Structure

Best_BG_performance/
│
├── activity_picker_correlation.ipynb
│   Selects cycling activities and correlates HR–P metrics with BG data.
│
├── intervals_nightscout_analysis.ipynb
│   Interval-based analysis with CGM overlays and performance metrics.
│
├── utilities/
│   Helper functions for data loading, preprocessing, and correlation logic.
│
└── output/
Generated plots, tables, and intermediate results.

Code

---

## 📘 Notebook Summaries

### **1. activity_picker_correlation.ipynb**
- Load and filter cycling activities  
- Extract HR–P pairs  
- Align timestamps with Nightscout CGM  
- Compute correlations between BG and HR–P efficiency  
- Visualize HR–P curve shifts across BG ranges  

### **2. intervals_nightscout_analysis.ipynb**
- Detect and segment intervals  
- Overlay BG traces during efforts  
- Identify performance degradation or improvement  
- Compute HR drift, power stability, and metabolic stress indicators  

---

## ⚙️ Installation


git clone https://github.com/grossato/Best_BG_performance
cd Best_BG_performance
Install dependencies:

bash
pip install numpy pandas matplotlib seaborn jupyter
pip install requests python-dateutil
pip install pynightscout
