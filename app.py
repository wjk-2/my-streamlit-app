import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from streamlit_option_menu import option_menu
from model_compatible import (
    CompatibleOccupationalStressModel as OccupationalStressModel, 
    CompatibleDepressionModel,
    process_user_input,
    process_depression_input
)
from data_manager import DataManager, create_data_management_page
MODEL_TYPE = "compatible"

st.set_page_config(
    page_title="职业健康风险智能分析系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '# 职业健康风险智能分析系统\n基于机器学习技术，评估职业紧张和抑郁症状风险'
    }
)

def local_css(file_name):
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(base_dir, file_name)
    
    try:
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"自定义样式文件未找到: {css_path}，使用默认样式")
    except UnicodeDecodeError:
        try:
            with open(css_path, encoding='gbk') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"无法读取样式文件: {e}")

def switch_to_page(page_name):
    """统一页面跳转逻辑，确保状态同步+即时生效"""
    st.session_state.page = page_name
    st.query_params = {"page": [page_name]}
    st.rerun()

def render_risk_result_panel(result: dict, risk_title: str):
    """统一风险结果展示样式，提升可读性与演示观感。"""
    risk_level = result.get('risk_level', '未知')
    prob = float(result.get('probability', 0))
    confidence = float(result.get('confidence', 0))

    color_map = {"低风险": "#2E7D32", "中风险": "#F9A825", "高风险": "#C62828"}
    badge_color = color_map.get(risk_level, "#6F42C1")
    st.markdown(
        f"""
        <div style="padding:14px 16px;border-radius:12px;background:linear-gradient(135deg,#f8f9fc,#eef1f8);border:1px solid #e6e9f2;">
            <div style="font-size:13px;color:#5f6472;">{risk_title}</div>
            <div style="margin-top:6px;display:inline-block;padding:4px 10px;border-radius:999px;background:{badge_color};color:#fff;font-weight:700;">
                {risk_level}
            </div>
            <div style="margin-top:10px;font-size:14px;color:#3d4250;">预测概率：<b>{prob:.1%}</b>　|　置信度：<b>{confidence:.1%}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.progress(int(max(0, min(100, round(prob * 100)))))

if 'page' not in st.session_state:
    st.session_state.page = "首页"

query_params = st.query_params
if 'page' in query_params and query_params['page'][0] in ["首页", "职业紧张预测", "抑郁症状预测", "数据分析", "数据管理", "关于系统"]:
    st.session_state.page = query_params['page'][0]

with st.sidebar:
    page_to_index = {
        "首页": 0,
        "职业紧张预测": 1,
        "抑郁症状预测": 2,
        "数据分析": 3,
        "数据管理": 4,
        "关于系统": 5
    }
    default_index = page_to_index.get(st.session_state.page, 0)
    
    selected = option_menu(
        menu_title="导航菜单",
        options=["首页", "职业紧张预测", "抑郁症状预测", "数据分析", "数据管理", "关于系统"],
        icons=["house", "clipboard-pulse", "heart-pulse", "bar-chart", "database", "info-circle"],
        menu_icon="cast",
        default_index=default_index, 
        key="nav_menu",
        styles={
            "container": {"padding": "5px", "background-color": "#6f42c1"},
            "icon": {"color": "white", "font-size": "18px"}, 
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#5a32a3"},
        }
    )

if selected != st.session_state.page:
    st.session_state.page = selected
    st.query_params = {"page": selected}
    st.rerun()

if selected == "首页":
    st.title("职业健康风险智能分析系统")
    st.markdown("### 基于多维度评估的职业健康风险智能分析系统")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>🧠 职业紧张预测</h3>
                <p>基于机器学习模型，输入职业和生活特征，预测职业紧张风险等级。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("开始预测", key="predict_btn"):
                st.session_state.page = "职业紧张预测"
                st.query_params = {"page": "职业紧张预测"}
                st.rerun()
    
    with col2:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>💗 抑郁症状预测</h3>
                <p>基于机器学习模型，评估抑郁症状风险等级，提供专业建议。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("抑郁评估", key="depression_btn"):
                st.session_state.page = "抑郁症状预测"
                st.query_params = {"page": "抑郁症状预测"}
                st.rerun()
    
    with col3:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>📊 数据分析</h3>
                <p>通过交互式图表和统计分析，深入了解职业健康的特征分布和影响因素。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("查看分析", key="analysis_btn"):
                st.session_state.page = "数据分析"
                st.query_params = {"page": "数据分析"}
                st.rerun()
    
    with col4:
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3>📋 数据管理</h3>
                <p>查看用户数据统计、导出分析报告和管理历史记录。</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("管理数据", key="data_management_btn"):
                st.session_state.page = "数据管理"
                st.query_params = {"page": "数据管理"}
                st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### ℹ️ 关于系统
    了解系统设计理念、技术架构和使用方法，获取更多帮助信息。
    """)
    if st.button("了解更多", key="about_btn"):
        st.session_state.page = "关于系统"
        st.query_params = {"page": "关于系统"}
        st.rerun()

elif selected == "职业紧张预测":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("职业紧张风险预测")
    with col2:
        if st.button("🏠 返回主页", key="back_from_predict"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    st.markdown("请输入以下职业紧张相关特征信息，系统将预测您的职业紧张风险等级")
    
    with st.form("职业紧张预测表单"):
        st.header("职业紧张特征输入")
        col1, col2 = st.columns(2)
        
        with col1:
            work_years = st.slider("工龄（年）", 0, 60, 5, key="stress_work_years")
            daily_work_hours = st.slider("日均工作时长（小时）", 0.0, 16.0, 8.0, 0.5, key="stress_daily_work_hours")
            daily_rest_hours = st.slider("工作休息时长（小时/天）", 0.0, 8.0, 1.0, 0.5, key="stress_daily_rest_hours")
            effort_reward = st.select_slider("付出与认可（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="stress_effort_reward")
            shift_tolerance = st.select_slider("轮班制承受度（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="stress_shift_tolerance")
            overtime = st.select_slider("加班情况（0无-3频繁）", options=[0, 1, 2, 3], value=1, key="stress_overtime_level")
        
        with col2:
            autonomy = st.select_slider("工作自由度（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="stress_autonomy")
            interest = st.select_slider("做事兴趣（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="stress_interest")
            self_harm = st.radio("自伤想法", ["无", "有"], horizontal=True, key="stress_self_harm")
            mood_impact = st.select_slider("情绪影响（0无-3严重）", options=[0, 1, 2, 3], value=1, key="stress_mood_impact")
            fatigue_acc = st.select_slider("疲劳蓄积等级（0无-3重度）", options=[0, 1, 2, 3], value=1, key="stress_fatigue_acc")
        
        submitted = st.form_submit_button("预测职业紧张风险", type="primary")

    if submitted:
        try:
            user_input = {
                '工龄': int(work_years),
                '日均工作时长': float(daily_work_hours),
                '工作休息时长': float(daily_rest_hours),
                '付出与认可': int(effort_reward),
                '轮班制承受度': int(shift_tolerance),
                '加班情况': int(overtime),
                '工作自由度': int(autonomy),
                '做事兴趣': int(interest),
                '自伤想法': 1 if self_harm == "有" else 0,
                '情绪影响': int(mood_impact),
                '疲劳蓄积等级': int(fatigue_acc),
            }
            
            model_input = user_input.copy()
            model = OccupationalStressModel()
            model.load_model()
            result = model.predict_stress(model_input)
            if MODEL_TYPE == "compatible":
                st.success("✅ 预测完成（使用兼容优化模型）")
            elif MODEL_TYPE == "optimized":
                st.success("✅ 预测完成（使用优化模型）")
            else:
                st.success("预测完成！")
            data_manager = DataManager()
            session_info = {
                'ip_address': 'unknown',
                'user_agent': 'unknown'
            }
            
            session_id = data_manager.save_user_record(user_input, result, session_info)
            if session_id:
                st.session_state.last_session_id = session_id
                st.session_state.last_prediction_result = result
                st.session_state.last_user_input = user_input
            
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'work_years': work_years,
                'daily_work_hours': daily_work_hours,
                'user_input': user_input,
                'risk_level': result['risk_level'],
                'probability': result['probability'],
                'confidence': result['confidence']
            }
            
            if result['risk_level'] == "高风险":
                report_data['suggestions'] = [
                    "寻求专业心理咨询服务",
                    "调整工作强度和时间安排",
                    "加强体育锻炼和健康管理",
                    "改善睡眠质量和饮食习惯",
                    "考虑与上级沟通工作压力问题"
                ]
            elif result['risk_level'] == "中风险":
                report_data['suggestions'] = [
                    "定期进行压力管理训练",
                    "保持工作与生活的平衡",
                    "培养积极的应对策略",
                    "关注身体信号，及时调整",
                    "加强社交支持和家庭沟通"
                ]
            else:
                report_data['suggestions'] = [
                    "继续保持健康的生活方式",
                    "定期进行自我压力评估",
                    "培养积极的心态和情绪管理",
                    "平衡工作和休息时间",
                    "持续关注身心健康指标"
                ]
            
            st.session_state.report_data = report_data
            
            st.subheader("预测结果")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                render_risk_result_panel(result, "职业紧张风险等级")
            
            with col2:
                try:
                    feature_importance = model.get_feature_importance()
                    st.write("**特征重要性排名:**")
                    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                        st.write(f"{i}. {feature}: {importance:.3f}")
                except:
                    st.info("🔧 特征重要性分析暂时不可用")
            
            st.subheader("个性化建议")
            if result['risk_level'] == "高风险":
                st.error("🔴 **高风险等级**")
                st.warning("""
                **立即行动建议:**
                - 寻求专业心理咨询服务
                - 调整工作强度和时间安排
                - 加强体育锻炼和健康管理
                - 改善睡眠质量和饮食习惯
                - 考虑与上级沟通工作压力问题
                """)
            elif result['risk_level'] == "中风险":
                st.warning("🟡 **中风险等级**")
                st.info("""
                **预防性建议:**
                - 定期进行压力管理训练
                - 保持工作与生活的平衡
                - 培养积极的应对策略
                - 关注身体信号，及时调整
                - 加强社交支持和家庭沟通
                """)
            else:
                st.success("🟢 **低风险等级**")
                st.info("""
                **维持良好状态建议:**
                - 继续保持健康的生活方式
                - 定期进行自我压力评估
                - 培养积极的心态和情绪管理
                - 平衡工作和休息时间
                - 持续关注身心健康指标
                """)
            
            st.subheader("📋 报告导出")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 生成PDF报告", key="generate_pdf"):
                    data_manager = DataManager()
                    pdf_filename = data_manager.export_report_pdf(
                        st.session_state.get('last_session_id', 'unknown'),
                        st.session_state.get('report_data', {})
                    )
                    if pdf_filename:
                        pdf_path = data_manager.get_report_file_path(pdf_filename)
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载PDF报告",
                                data=f,
                                file_name=pdf_filename,
                                mime="application/pdf"
                            )
            
            with col2:
                if st.button("📊 导出CSV数据", key="export_csv"):
                    data_manager = DataManager()
                    csv_filename = data_manager.export_report_csv(
                        st.session_state.get('last_session_id', 'unknown'),
                        st.session_state.get('report_data', {})
                    )
                    if csv_filename:
                        csv_path = data_manager.get_report_file_path(csv_filename)
                        with open(csv_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载CSV数据",
                                data=f,
                                file_name=csv_filename,
                                mime="text/csv"
                            )
                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            st.info("系统将使用模拟数据进行预测...")
            risk_level = np.random.choice(["低风险", "中风险", "高风险"], p=[0.4, 0.4, 0.2])
            st.metric("职业紧张风险等级", risk_level)

elif selected == "抑郁症状预测":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("抑郁症状风险预测")
    with col2:
        if st.button("🏠 返回主页", key="back_from_depression"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    st.markdown("请输入以下抑郁症状相关特征信息，系统将预测您的抑郁症状风险等级")
    
    with st.form("抑郁症状预测表单"):
        st.header("抑郁症状特征输入")
        col1, col2 = st.columns(2)
        
        with col1:
            local_residence_years = st.slider("本地居住时间（年）", 0, 80, 5, key="dep_local_residence_years")
            labor_type = st.selectbox(
                "劳动确定方式",
                options=[0, 1, 2, 3],
                format_func=lambda x: {0: "不确定/临时", 1: "短期合同", 2: "长期合同", 3: "其他"}.get(x, str(x)),
                key="dep_labor_type",
            )
            work_years = st.slider("工龄（年）", 0, 60, 5, key="dep_work_years")
            daily_work_hours = st.slider("日均工作时长（小时）", 0.0, 16.0, 8.0, 0.5, key="dep_daily_work_hours")
            daily_rest_hours = st.slider("工作休息时长（小时/天）", 0.0, 8.0, 1.0, 0.5, key="dep_daily_rest_hours")
            night_shift_count = st.slider("夜班次数（近1月）", 0, 31, 0, key="dep_night_shift_count")
            effort_reward = st.select_slider("付出与认可（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="dep_effort_reward")
            shift_tolerance = st.select_slider("轮班制承受度（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="dep_shift_tolerance")
            overtime = st.select_slider("加班情况（0无-3频繁）", options=[0, 1, 2, 3], value=1, key="dep_overtime_level")
            autonomy = st.select_slider("工作自由度（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="dep_autonomy")
            interest = st.select_slider("做事兴趣（1低-5高）", options=[1, 2, 3, 4, 5], value=3, key="dep_interest")
        
        with col2:
            self_harm = st.radio("自伤想法", ["无", "有"], horizontal=True, key="dep_self_harm")
            mood_impact = st.select_slider("情绪影响（0无-3严重）", options=[0, 1, 2, 3], value=1, key="dep_mood_impact")
            smoking_amount = st.slider("吸烟量（支/天）", 0, 60, 0, key="dep_smoking_amount")
            fried_freq = st.select_slider("油炸摄入频率（0从不-5每天）", options=[0, 1, 2, 3, 4, 5], value=2, key="dep_fried_freq")
            dessert_freq = st.select_slider("甜点摄入频率（0从不-5每天）", options=[0, 1, 2, 3, 4, 5], value=2, key="dep_dessert_freq")
            sleep_time = st.slider("入睡时间（24小时制，小时）", 0, 23, 23, key="dep_sleep_time")
            hard_to_sleep = st.radio("难入睡", ["否", "是"], horizontal=True, key="dep_hard_to_sleep")
            wake_early = st.radio("醒太早", ["否", "是"], horizontal=True, key="dep_wake_early")
            health_status = st.select_slider("健康状况（1差-5好）", options=[1, 2, 3, 4, 5], value=3, key="dep_health_status")
            sick_leave = st.slider("因病休工（近1年天数）", 0, 365, 0, key="dep_sick_leave")
            fatigue_acc = st.select_slider("疲劳蓄积等级（0无-3重度）", options=[0, 1, 2, 3], value=1, key="dep_fatigue_acc")
        
        submitted = st.form_submit_button("预测抑郁症状风险", type="primary")

    if submitted:
        try:
            user_input = {
                '本地居住时间': int(local_residence_years),
                '劳动确定方式': int(labor_type),
                '工龄': int(work_years),
                '日均工作时长': float(daily_work_hours),
                '工作休息时长': float(daily_rest_hours),
                '夜班次数': int(night_shift_count),
                '付出与认可': int(effort_reward),
                '轮班制承受度': int(shift_tolerance),
                '加班情况': int(overtime),
                '工作自由度': int(autonomy),
                '做事兴趣': int(interest),
                '自伤想法': 1 if self_harm == "有" else 0,
                '情绪影响': int(mood_impact),
                '吸烟量': int(smoking_amount),
                '油炸摄入频率': int(fried_freq),
                '甜点摄入频率': int(dessert_freq),
                '入睡时间': int(sleep_time),
                '难入睡': 1 if hard_to_sleep == "是" else 0,
                '醒太早': 1 if wake_early == "是" else 0,
                '健康状况': int(health_status),
                '因病休工': int(sick_leave),
                '疲劳蓄积等级': int(fatigue_acc),
            }
            
            model_input = process_depression_input(user_input)
            model = CompatibleDepressionModel()
            model.load_model()
            result = model.predict_depression(model_input)
            
            st.success("✅ 抑郁症状风险预测完成！")
            
            data_manager = DataManager()
            session_info = {
                'ip_address': 'unknown',
                'user_agent': 'unknown',
                'prediction_type': 'depression'
            }
            
            session_id = data_manager.save_depression_record(user_input, result, session_info)
            if session_id:
                st.session_state.last_session_id = session_id
                st.session_state.last_prediction_result = result
                st.session_state.last_user_input = user_input
            
            report_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_input': user_input,
                'risk_level': result['risk_level'],
                'probability': result['probability'],
                'confidence': result['confidence'],
                'prediction_type': 'depression'
            }
            
            if result['risk_level'] == "高风险":
                report_data['suggestions'] = [
                    "寻求专业心理医生或精神科医生的帮助",
                    "立即预约心理健康评估",
                    "与信任的人分享您的感受",
                    "尽量保持规律的作息和饮食",
                    "避免做出重大决策或处理复杂问题"
                ]
            elif result['risk_level'] == "中风险":
                report_data['suggestions'] = [
                    "考虑咨询心理健康专业人士",
                    "尝试进行放松训练和冥想",
                    "保持适度运动和社交活动",
                    "记录情绪变化以便自我监控",
                    "学习压力管理技巧"
                ]
            else:
                report_data['suggestions'] = [
                    "继续保持健康的生活方式",
                    "定期进行心理健康自查",
                    "维持良好的人际关系",
                    "保持规律的运动和作息",
                    "关注心理健康知识"
                ]
            
            st.session_state.report_data = report_data
            
            st.subheader("预测结果")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                render_risk_result_panel(result, "抑郁症状风险等级")
            
            with col2:
                try:
                    feature_importance = model.get_feature_importance()
                    st.write("**特征重要性排名:**")
                    for i, (feature, importance) in enumerate(feature_importance[:5], 1):
                        st.write(f"{i}. {feature}: {importance:.3f}")
                except:
                    st.info("🔧 特征重要性分析暂时不可用")
            
            st.subheader("个性化建议")
            if result['risk_level'] == "高风险":
                st.error("🔴 **高风险等级**")
                st.warning("""
                **立即行动建议:**
                - 寻求专业心理医生或精神科医生的帮助
                - 立即预约心理健康评估
                - 与信任的人分享您的感受
                - 尽量保持规律的作息和饮食
                - 避免做出重大决策或处理复杂问题
                """)
            elif result['risk_level'] == "中风险":
                st.warning("🟡 **中风险等级**")
                st.info("""
                **预防性建议:**
                - 考虑咨询心理健康专业人士
                - 尝试进行放松训练和冥想
                - 保持适度运动和社交活动
                - 记录情绪变化以便自我监控
                - 学习压力管理技巧
                """)
            else:
                st.success("🟢 **低风险等级**")
                st.info("""
                **维持良好状态建议:**
                - 继续保持健康的生活方式
                - 定期进行心理健康自查
                - 维持良好的人际关系
                - 保持规律的运动和作息
                - 关注心理健康知识
                """)
            
            st.subheader("📋 报告导出")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 生成报告", key="generate_depression_report"):
                    data_manager = DataManager()
                    report_filename = data_manager.export_depression_report(
                        st.session_state.get('last_session_id', 'unknown'),
                        st.session_state.get('report_data', {})
                    )
                    if report_filename:
                        report_path = data_manager.get_report_file_path(report_filename)
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载报告",
                                data=f,
                                file_name=report_filename,
                                mime="text/csv"
                            )
            
            with col2:
                if st.button("📊 导出CSV数据", key="export_depression_csv"):
                    data_manager = DataManager()
                    csv_filename = data_manager.export_depression_csv(
                        st.session_state.get('last_session_id', 'unknown'),
                        st.session_state.get('report_data', {})
                    )
                    if csv_filename:
                        csv_path = data_manager.get_report_file_path(csv_filename)
                        with open(csv_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载CSV数据",
                                data=f,
                                file_name=csv_filename,
                                mime="text/csv"
                            )
                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            st.info("系统将使用模拟数据进行预测...")
            risk_level = np.random.choice(["低风险", "中风险", "高风险"], p=[0.4, 0.4, 0.2])
            st.metric("抑郁症状风险等级", risk_level)

elif selected == "数据分析":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("数据分析")
    with col2:
        if st.button("🏠 返回主页", key="back_from_analysis"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    
    try:
        from analysis import create_analysis_page
        create_analysis_page()
    except Exception as e:
        st.error(f"数据分析模块加载失败: {e}")
        st.info("正在使用简化版数据分析功能...")
        
        try:
            df = pd.read_excel('9066新业态数据-特征工程后.xlsx')
            st.success("数据加载成功！")
            
            st.subheader("📊 基础统计分析")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总样本数", len(df))
            
            with col2:
                stress_rate = df['是否职业紧张'].mean()
                st.metric("职业紧张比例", f"{stress_rate:.1%}")
            
            with col3:
                if '日均工作时长' in df.columns:
                    avg_hours = df['日均工作时长'].mean()
                    st.metric("日均工作时长", f"{avg_hours:.1f}小时")
                elif '周均工作时间' in df.columns:
                    avg_hours = df['周均工作时间'].mean()
                    st.metric("周均工作时间", f"{avg_hours:.1f}小时")
                else:
                    st.metric("工作时长", "N/A")
            
        except Exception as data_error:
            st.error(f"数据加载错误: {data_error}")


elif selected == "数据管理":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("数据管理与报告导出")

    with col2:
        if st.button("🏠 返回主页", key="back_from_data_management"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    
    try:
        create_data_management_page()
    except Exception as e:
        st.error(f"数据管理模块加载失败: {e}")
        st.info("数据管理功能正在开发中...")

elif selected == "关于系统":
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("关于职业健康风险智能分析系统")
    with col2:
        if st.button("🏠 返回主页", key="back_from_about"):
            st.session_state.page = "首页"
            st.query_params = {"page": "首页"}
            st.rerun()
    st.markdown("""
    ### 系统简介
    本系统基于机器学习技术，通过对职业和生活特征的分析，预测个体的职业紧张风险和抑郁症状风险等级。
    
    ### 主要功能
    - **职业紧张预测**：输入个人特征，预测职业紧张风险
    - **抑郁症状预测**：评估抑郁症状风险等级，提供专业建议
    - **数据分析**：展示职业健康相关数据的统计分析
    - **个性化建议**：根据预测结果提供针对性的改善建议
    
    ### 技术特点
    - 基于大数据集训练（3046条样本）
    - 采用先进的随机森林机器学习算法
    - 提供直观的可视化结果
    - 支持风险因素分析
    """)

local_css("assets/style.css")

def main():
    """应用主入口函数"""
    # Streamlit应用已在顶层执行，此处仅为封装提供入口点
    pass

if __name__ == "__main__":
    # 应用在顶层已启动，此块仅用于模块化封装
    pass