import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, data_path='9066新业态数据-特征工程后.xlsx'):
        self.data_path = data_path
        self.df = None
        self.load_data()
        
    def load_data(self):
        """加载数据"""
        try:
            self.df = pd.read_excel(self.data_path)
            st.success("数据加载成功！")
        except Exception as e:
            st.error(f"数据加载失败: {e}")
            self.create_sample_data()

    def _apply_chart_style(self, fig, title_left=True):
        """统一图表风格，提升整体观感"""
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor='rgba(248,249,252,0.9)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=13),
            margin=dict(l=24, r=24, t=62, b=24),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        if title_left:
            fig.update_layout(title=dict(x=0.02, xanchor='left'))
        return fig
    
    def create_sample_data(self):
        """创建模拟数据用于演示"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            '年龄': np.random.normal(35, 10, n_samples),
            '工龄': np.random.normal(8, 6, n_samples),
            '周均工作时间': np.random.normal(45, 8, n_samples),
            '疲劳程度分级': np.random.randint(0, 4, n_samples),
            '收入水平': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
            '是否职业紧张': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            '是否抑郁症状': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            '是否睡眠障碍': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            '性别': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            '教育程度': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.4, 0.1])
        }
        
        self.df = pd.DataFrame(data)
        st.info("使用示例数据进行分析")
    
    def get_data_overview(self):
        """数据概览统计"""
        st.subheader("📊 数据概览")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总样本数", len(self.df))
        
        with col2:
            if '是否职业紧张' in self.df.columns:
                st.metric("职业紧张比例", f"{self.df['是否职业紧张'].mean():.1%}")
            else:
                st.metric("职业紧张比例", "N/A")
        
        with col3:
            if '年龄' in self.df.columns:
                avg_age = self.df['年龄'].mean()
                st.metric("平均年龄", f"{avg_age:.1f}岁")
            elif '工龄' in self.df.columns:
                avg_work_years = self.df['工龄'].mean()
                st.metric("平均工龄", f"{avg_work_years:.1f}年")
            else:
                st.metric("平均年龄/工龄", "N/A")
        
        with col4:
            if '周均工作时间' in self.df.columns:
                avg_hours = self.df['周均工作时间'].mean()
                st.metric("周均工作时间", f"{avg_hours:.1f}小时")
            elif '日均工作时长' in self.df.columns:
                avg_daily_hours = self.df['日均工作时长'].mean()
                st.metric("日均工作时长", f"{avg_daily_hours:.1f}小时")
            else:
                st.metric("工作时长", "N/A")

        st.write("**数据预览:**")
        st.dataframe(self.df.head(), use_container_width=True)
    
    def create_feature_distribution_charts(self):
        """创建特征分布图表"""
        st.subheader("📈 特征分布分析")
        preferred_features = [
            # 职业紧张新特征
            '工龄', '日均工作时长', '工作休息时长', '付出与认可', '轮班制承受度', '加班情况',
            '工作自由度', '做事兴趣', '自伤想法', '情绪影响', '疲劳蓄积等级',
            # 抑郁症状新特征（补充）
            '本地居住时间', '劳动确定方式', '夜班次数', '吸烟量', '油炸摄入频率', '甜点摄入频率',
            '入睡时间', '难入睡', '醒太早', '健康状况', '因病休工',
            # 兼容旧数据常见列
            '年龄', '本岗位工龄', '周均工作时间', '日均加班时间', '疲劳程度分级',
        ]
        feature_options = [f for f in preferred_features if f in self.df.columns]
        if not feature_options:
            st.info("当前数据集中没有可用于分布分析的特征列。")
            return
        selected_feature = st.selectbox("选择要分析的特征", feature_options)
        
        if selected_feature:
            col1, col2 = st.columns(2)
            
            with col1:
                # 直方图
                fig = px.histogram(self.df, x=selected_feature, 
                                 title=f'{selected_feature}分布',
                                 nbins=24,
                                 color_discrete_sequence=['#7C4DFF'])
                fig.update_traces(
                    marker_line_color='#FFFFFF',
                    marker_line_width=1.2,
                    opacity=0.9,
                    hovertemplate=f'{selected_feature}: %{{x}}<br>频数: %{{y}}<extra></extra>'
                )
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='rgba(248,249,252,0.9)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=13),
                    title=dict(x=0.02, xanchor='left'),
                    margin=dict(l=20, r=20, t=55, b=20),
                    bargap=0.08,
                    yaxis_title='频数',
                    xaxis_title=selected_feature,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # 箱线图
                fig = px.box(self.df, y=selected_feature, 
                           title=f'{selected_feature}箱线图',
                           color_discrete_sequence=['#6f42c1'])
                st.plotly_chart(fig, use_container_width=True)
    
    def create_correlation_analysis(self):
        """相关性分析"""
        st.subheader("🔗 相关性分析")
        
        # 优先使用“新特征”，不存在则自动回退为数据集中可用的数值列
        candidate = [
            '工龄', '日均工作时长', '工作休息时长', '付出与认可', '轮班制承受度', '加班情况',
            '工作自由度', '做事兴趣', '自伤想法', '情绪影响', '疲劳蓄积等级',
            '本地居住时间', '劳动确定方式', '夜班次数', '吸烟量', '油炸摄入频率', '甜点摄入频率',
            '入睡时间', '难入睡', '醒太早', '健康状况', '因病休工',
            # 旧字段
            '年龄', '本岗位工龄', '周均工作时间', '日均加班时间', '疲劳程度分级', '饮酒量',
        ]
        numeric_features = [f for f in candidate if f in self.df.columns]
        if not numeric_features:
            numeric_features = list(self.df.select_dtypes(include=['int64', 'float64']).columns)
        if len(numeric_features) < 2:
            st.info("可用于相关性分析的数值特征不足。")
            return
        corr_df = self.df[numeric_features].corr()
        
        # 相关性热力图
        fig = px.imshow(corr_df,
                       title="特征相关性热力图",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        fig = self._apply_chart_style(fig)
        fig.update_layout(
            coloraxis_colorbar=dict(title='相关系数'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
        )
        fig.update_traces(
            hovertemplate='X: %{x}<br>Y: %{y}<br>相关系数: %{z:.3f}<extra></extra>'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_group_analysis(self):
        """分组分析"""
        st.subheader("👥 分组对比分析")
        
        # 分组选项：优先使用存在的“分类/离散”字段
        group_candidates = [
            '劳动确定方式', '加班情况', '难入睡', '醒太早', '自伤想法',
            # 兼容旧数据
            '收入水平', '教育程度', '性别', '是否夜班', '是否吸烟', '是否饮酒', '是否睡眠障碍'
        ]
        group_options = [g for g in group_candidates if g in self.df.columns]
        if not group_options:
            st.info("当前数据集中缺少可用的分组变量。")
            return
        group_by = st.selectbox("按变量分组", group_options)

        metric_candidates = [
            '工龄', '日均工作时长', '工作休息时长', '付出与认可', '轮班制承受度', '工作自由度',
            '做事兴趣', '情绪影响', '疲劳蓄积等级', '夜班次数', '吸烟量', '油炸摄入频率', '甜点摄入频率',
            '入睡时间', '健康状况', '因病休工',
            # 旧字段
            '年龄', '本岗位工龄', '周均工作时间', '日均加班时间', '疲劳程度分级',
        ]
        metric_options = [m for m in metric_candidates if m in self.df.columns]
        if not metric_options:
            st.info("当前数据集中缺少可用的分析指标。")
            return
        metric = st.selectbox("分析指标", metric_options)
        
        if group_by and metric:
            if group_by not in self.df.columns or len(self.df[group_by].dropna().unique()) < 2:
                st.warning(f"⚠️ 无法进行分组分析：变量 '{group_by}' 不存在或分组数量不足")
                return
            
            try:
                # 分组统计
                grouped_stats = self.df.groupby(group_by)[metric].agg(['mean', 'std', 'count'])
                
                # 分组柱状图
                fig = px.bar(grouped_stats.reset_index(), 
                            x=group_by, y='mean',
                            title=f'{metric}按{group_by}分组对比',
                            color=group_by,
                            color_discrete_sequence=px.colors.qualitative.Set3)
                # 手动设置轴标签，解决"意思"标签问题
                fig.update_layout(
                    xaxis_title=group_by,
                    yaxis_title=metric
                )
                fig = self._apply_chart_style(fig)
                fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                
                # 分组箱线图
                fig = px.box(self.df, x=group_by, y=metric,
                            title=f'{metric}分布（按{group_by}分组）',
                            color=group_by,
                            color_discrete_sequence=px.colors.qualitative.Set3,  # 使用与柱状图一致的颜色
                            category_orders={group_by: sorted(self.df[group_by].unique())})  # 确保分组顺序一致
                # 手动设置轴标签
                fig.update_layout(
                    xaxis_title=group_by,
                    yaxis_title=metric
                )
                fig = self._apply_chart_style(fig)
                fig.update_traces(
                    line_width=1.2,
                    hovertemplate=f'{group_by}: %{{x}}<br>{metric}: %{{y:.2f}}<extra></extra>'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"分组分析过程中出现错误: {e}")
                st.info("系统将显示基础统计分析作为替代")
                st.write(f"**{metric} 描述性统计:**")
                st.write(self.df[metric].describe())
    
    def create_stress_risk_analysis(self):
        """职业紧张和抑郁症状风险分析"""
        st.subheader("⚠️ 职业健康风险因素分析")
        
        # 职业紧张风险分析
        analysis_type = st.selectbox("选择分析类型", ["职业紧张风险", "抑郁症状风险"])
        
        if analysis_type == "职业紧张风险":
            target_col = '是否职业紧张'
            title_prefix = '职业紧张'
        else:
            target_col = '是否抑郁症状'
            title_prefix = '抑郁症状'
        
        variables = [
            # 新特征里更可能有解释性的离散变量
            '劳动确定方式', '加班情况', '难入睡', '醒太早', '自伤想法',
            # 兼容旧字段
            '收入水平', '教育程度', '性别', '是否夜班', '是否睡眠障碍'
        ]
        
        for var in variables:
            if var not in self.df.columns:
                continue
            risk_rates = self.df.groupby(var)[target_col].mean().reset_index()
            
            fig = px.bar(risk_rates, x=var, y=target_col,
                        title=f'{var}与{title_prefix}关系',
                        color=var,
                        color_discrete_sequence=px.colors.qualitative.Bold)
            # 手动设置轴标签
            fig.update_layout(
                xaxis_title=var,
                yaxis_title=f'{title_prefix}比例'
            )
            fig = self._apply_chart_style(fig)
            fig.update_layout(yaxis_tickformat='.0%')
            fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
    def create_insights(self):
        """数据分析洞察"""
        st.subheader("💡 数据洞察")
        
        insights = []
        
        # 职业紧张分析
        if '是否职业紧张' in self.df.columns:
            stress_rate = self.df['是否职业紧张'].mean()
            if stress_rate > 0.3:
                insights.append(f"⚠️ **高风险预警**: 总体职业紧张比例达到{stress_rate:.1%}，需要关注")
        
        # 抑郁症状分析
        if '是否抑郁症状' in self.df.columns:
            depression_rate = self.df['是否抑郁症状'].mean()
            if depression_rate > 0.2:
                insights.append(f"⚠️ **抑郁症状预警**: 总体抑郁症状比例达到{depression_rate:.1%}，需要关注")
        
        # 工作时间与疲劳关系
        if '周均工作时间' in self.df.columns and '疲劳程度分级' in self.df.columns:
            work_fatigue_corr = self.df['周均工作时间'].corr(self.df['疲劳程度分级'])
            if abs(work_fatigue_corr) > 0.1:
                direction = "正相关" if work_fatigue_corr > 0 else "负相关"
                insights.append(f"📊 **工作时间影响**: 周工作时间与疲劳程度存在{direction}关系 (r={work_fatigue_corr:.2f})")
        
        # 收入水平与紧张关系
        if '收入水平' in self.df.columns and '是否职业紧张' in self.df.columns:
            income_stress = self.df.groupby('收入水平')['是否职业紧张'].mean()
            if len(income_stress) >= 2:
                if income_stress.iloc[0] > income_stress.iloc[-1]:
                    insights.append("?? **收入影响**: 低收入群体的职业紧张风险更高")

        for insight in insights:
            st.info(insight)

        st.write("**关键统计指标:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if '疲劳程度分级' in self.df.columns:
                avg_fatigue = self.df['疲劳程度分级'].mean()
                st.metric("平均疲劳程度", f"{avg_fatigue:.1f}/3")
            else:
                st.metric("平均疲劳程度", "N/A")
        
        with col2:
            if '是否职业紧张' in self.df.columns:
                stress_rate = self.df['是否职业紧张'].mean()
                st.metric("职业紧张比例", f"{stress_rate:.1%}")
            else:
                st.metric("职业紧张比例", "N/A")
        
        with col3:
            if '周均工作时间' in self.df.columns:
                avg_work_hours = self.df['周均工作时间'].mean()
                st.metric("平均周工作时间", f"{avg_work_hours:.1f}小时")
            elif '日均工作时长' in self.df.columns:
                avg_daily_work_hours = self.df['日均工作时长'].mean()
                st.metric("平均日工作时长", f"{avg_daily_work_hours:.1f}小时")
            else:
                st.metric("平均周工作时间", "N/A")

        # 新增洞察可视化：双风险比例横向对比，便于演示展示
        risk_summary = []
        if '是否职业紧张' in self.df.columns:
            risk_summary.append({'风险类型': '职业紧张', '阳性比例': float(self.df['是否职业紧张'].mean())})
        if '是否抑郁症状' in self.df.columns:
            risk_summary.append({'风险类型': '抑郁症状', '阳性比例': float(self.df['是否抑郁症状'].mean())})

        if risk_summary:
            summary_df = pd.DataFrame(risk_summary)
            fig = px.bar(
                summary_df,
                x='风险类型',
                y='阳性比例',
                color='风险类型',
                title='关键风险比例总览',
                color_discrete_sequence=['#7C4DFF', '#26A69A']
            )
            fig = self._apply_chart_style(fig)
            fig.update_layout(
                xaxis_title='风险类型',
                yaxis_title='阳性比例',
                yaxis_tickformat='.0%'
            )
            fig.update_traces(texttemplate='%{y:.1%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

def create_analysis_page():
    """创建数据分析页面"""

    st.title("📊 数据分析")
    st.markdown("深入探索职业紧张相关数据的统计分析和可视化洞察")

    analyzer = DataAnalyzer()
    
    if analyzer.df is not None:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 数据概览", "📈 特征分布", "🔗 相关性", "👥 分组分析", "💡 数据洞察"
        ])
        
        with tab1:
            analyzer.get_data_overview()
        
        with tab2:
            analyzer.create_feature_distribution_charts()
        
        with tab3:
            analyzer.create_correlation_analysis()
        
        with tab4:
            analyzer.create_group_analysis()
        
        with tab5:
            analyzer.create_stress_risk_analysis()
            analyzer.create_insights()
    
    else:
        st.error("无法加载数据文件")

if __name__ == "__main__":
    create_analysis_page()