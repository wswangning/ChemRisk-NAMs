ChemRisk-NAMs

基于PBPK-TRAEC-AIVIVE的化学品健康风险评估系统**

简介

ChemRisk-NAMs 是一个集成框架，用于在数据极度有限的条件下对化学品进行人体健康风险评估。系统融合了：

PBPK模型（基于生理的药代动力学模拟）
QIVIVE（定量体外-体内外推）
AIVIVE（机器学习增强的毒性通路预测）
TRAEC（证据整合、不确定性量化与风险决策）

安装

确保以下文件位于项目根目录：
aivive_xgboost_model.pkl（模型文件）
安装依赖：
bash
pip install streamlit pandas numpy matplotlib plotly requests joblib rdkit-pypi xgboost
启动应用：
bash
streamlit run app.py
该版本已修复所有已知错误，并实现了基于真实结构的 AIVIVE 预测。测试时请勾选“连接实时数据库”，输入任意真实 CAS（如 58-08-2）验证。
