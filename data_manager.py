import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import os
import json
import uuid
from typing import Dict, List, Any
import plotly.express as px

class DataManager:
    """用户数据管理和报告导出模块"""
    
    def __init__(self, data_dir='user_data'):
        self.data_dir = data_dir
        self.user_data_file = os.path.join(data_dir, 'user_records.csv')
        self.reports_dir = os.path.join(data_dir, 'reports')
        self.create_directories()
        self.initialize_user_data()
    
    def create_directories(self):
        """创建必要的目录结构"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def initialize_user_data(self):
        """初始化用户数据文件"""
        if not os.path.exists(self.user_data_file):
            df = pd.DataFrame(columns=self._get_base_columns())
            df.to_csv(self.user_data_file, index=False, encoding='utf-8')

    def _get_base_columns(self) -> List[str]:
        return [
            'user_id',
            'session_id',
            'timestamp',
            'prediction_type',
            'risk_level',
            'risk_probability',
            'confidence',
            'user_input_json',
            'ip_address',
            'user_agent',
        ]

    def _append_record_csv(self, csv_path: str, record: Dict[str, Any]) -> None:
        """以“列自动扩展”的方式追加记录，兼容字段变更与旧文件。"""
        new_df = pd.DataFrame([record])
        if os.path.exists(csv_path):
            old_df = pd.read_csv(csv_path)
            all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
            old_df = old_df.reindex(columns=all_cols)
            new_df = new_df.reindex(columns=all_cols)
            updated = pd.concat([old_df, new_df], ignore_index=True)
        else:
            updated = new_df
        updated.to_csv(csv_path, index=False, encoding='utf-8')
    
    def save_user_record(self, user_input: Dict, prediction_result: Dict, 
                        session_info: Dict = None) -> str:
        """保存用户输入和预测结果"""
        try:
            user_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            record = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'prediction_type': (session_info or {}).get('prediction_type', 'stress'),
                'risk_probability': prediction_result.get('probability', 0),
                'risk_level': prediction_result.get('risk_level', '未知'),
                'confidence': prediction_result.get('confidence', 0),
                'user_input_json': json.dumps(user_input, ensure_ascii=False),
                'ip_address': session_info.get('ip_address', 'unknown') if session_info else 'unknown',
                'user_agent': session_info.get('user_agent', 'unknown') if session_info else 'unknown'
            }
            self._append_record_csv(self.user_data_file, record)
            
            detailed_record = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'user_input': user_input,
                'prediction_result': prediction_result,
                'session_info': session_info or {}
            }
            
            json_filename = f"{session_id}.json"
            json_path = os.path.join(self.reports_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_record, f, ensure_ascii=False, indent=2)
            
            return session_id
            
        except Exception as e:
            st.error(f"保存用户记录时出错: {e}")
            return None
    
    def get_user_statistics(self) -> Dict:
        """获取用户数据统计信息"""
        try:
            if not os.path.exists(self.user_data_file):
                return {}
            
            df = pd.read_csv(self.user_data_file)
            
            if df.empty:
                return {}
            
            stats = {
                'total_users': df['user_id'].nunique(),
                'total_sessions': len(df),
                'today_sessions': len(df[df['timestamp'].str.startswith(datetime.now().strftime('%Y-%m-%d'))]),
                'risk_distribution': df['risk_level'].value_counts().to_dict(),
                'most_common_risk': df['risk_level'].mode().iloc[0] if not df['risk_level'].mode().empty else '未知'
            }
            
            return stats
            
        except Exception as e:
            st.error(f"获取统计信息时出错: {e}")
            return {}
    
    def export_report_pdf(self, session_id: str, report_data: Dict) -> str:
        """生成报告文件（当前以文本形式导出，便于打包/跨平台）"""
        try:
            title = "职业健康风险分析报告"
            report_content = f"""{title}
======================

报告时间: {report_data.get('timestamp', '未知')}
会话ID: {session_id}

预测结果:
---------
风险等级: {report_data.get('risk_level', '未知')}
预测概率: {report_data.get('probability', 0)*100:.1f}%
置信度: {report_data.get('confidence', 0)*100:.1f}%

关键输入特征（如有）:
--------------------
{json.dumps(report_data.get('user_input', {}), ensure_ascii=False, indent=2)}

健康建议:
---------
"""
            
            suggestions = report_data.get('suggestions', [])
            for i, suggestion in enumerate(suggestions, 1):
                report_content += f"\n{i}. {suggestion}"
            
            # 保存为文本文件
            txt_filename = f"report_{session_id}.txt"
            txt_path = os.path.join(self.reports_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return txt_filename
            
        except Exception as e:
            st.error(f"生成报告时出错: {e}")
            return None
    
    def export_report_csv(self, session_id: str, data: Dict) -> str:
        """导出CSV格式的报告数据"""
        try:
            csv_filename = f"report_{session_id}.csv"
            csv_path = os.path.join(self.reports_dir, csv_filename)
            
            # 创建CSV数据
            csv_data = {
                '项目': ['评估时间', '风险等级', '预测概率', '置信度'],
                '数值': [
                    data.get('timestamp', '未知'),
                    data.get('risk_level', '未知'),
                    f"{data.get('probability', 0)*100:.1f}%",
                    f"{data.get('confidence', 0)*100:.1f}%",
                ]
            }
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            return csv_filename
            
        except Exception as e:
            st.error(f"导出CSV报告时出错: {e}")
            return None
    
    def get_report_file_path(self, filename: str) -> str:
        """获取报告文件的完整路径"""
        return os.path.join(self.reports_dir, filename)
    
    def cleanup_old_reports(self, days_old: int = 30):
        """清理超过指定天数的旧报告文件"""
        try:
            cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
            
            for filename in os.listdir(self.reports_dir):
                file_path = os.path.join(self.reports_dir, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if file_time < cutoff_time:
                        os.remove(file_path)
                        
        except Exception as e:
            st.warning(f"清理旧报告时出错: {e}")

    def save_depression_record(self, user_input: Dict, prediction_result: Dict, 
                                session_info: Dict = None) -> str:
        """保存抑郁症状预测用户记录"""
        try:
            user_id = str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            record = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'prediction_type': 'depression',
                'risk_probability': prediction_result.get('probability', 0),
                'risk_level': prediction_result.get('risk_level', '未知'),
                'confidence': prediction_result.get('confidence', 0),
                'user_input_json': json.dumps(user_input, ensure_ascii=False),
                'ip_address': session_info.get('ip_address', 'unknown') if session_info else 'unknown',
                'user_agent': session_info.get('user_agent', 'unknown') if session_info else 'unknown'
            }
            
            depression_file = os.path.join(self.data_dir, 'depression_records.csv')
            
            self._append_record_csv(depression_file, record)
            
            detailed_record = {
                'user_id': user_id,
                'session_id': session_id,
                'timestamp': timestamp,
                'prediction_type': 'depression',
                'user_input': user_input,
                'prediction_result': prediction_result,
                'session_info': session_info or {}
            }
            
            json_filename = f"depression_{session_id}.json"
            json_path = os.path.join(self.reports_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_record, f, ensure_ascii=False, indent=2)
            
            return session_id
            
        except Exception as e:
            st.error(f"保存抑郁症状记录时出错: {e}")
            return None
    
    def export_depression_report(self, session_id: str, report_data: Dict) -> str:
        """生成抑郁症状分析报告"""
        try:
            report_content = f"""抑郁症状风险分析报告
=====================

报告时间: {report_data.get('timestamp', '未知')}
会话ID: {session_id}

预测结果:
---------
风险等级: {report_data.get('risk_level', '未知')}
预测概率: {report_data.get('probability', 0)*100:.1f}%
置信度: {report_data.get('confidence', 0)*100:.1f}%

关键输入特征（如有）:
--------------------
{json.dumps(report_data.get('user_input', {}), ensure_ascii=False, indent=2)}

健康建议:
---------
"""
            
            suggestions = report_data.get('suggestions', [])
            for i, suggestion in enumerate(suggestions, 1):
                report_content += f"\n{i}. {suggestion}"
            
            # 保存为文本文件
            txt_filename = f"depression_report_{session_id}.txt"
            txt_path = os.path.join(self.reports_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            return txt_filename
            
        except Exception as e:
            st.error(f"生成抑郁症状报告时出错: {e}")
            return None
    
    def export_depression_csv(self, session_id: str, data: Dict) -> str:
        """导出抑郁症状CSV格式的报告数据"""
        try:
            csv_filename = f"depression_report_{session_id}.csv"
            csv_path = os.path.join(self.reports_dir, csv_filename)
            
            # 创建CSV数据
            csv_data = {
                '项目': ['评估时间', '风险等级', '预测概率', '置信度'],
                '数值': [
                    data.get('timestamp', '未知'),
                    data.get('risk_level', '未知'),
                    f"{data.get('probability', 0)*100:.1f}%",
                    f"{data.get('confidence', 0)*100:.1f}%",
                ]
            }
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            return csv_filename
            
        except Exception as e:
            st.error(f"导出抑郁症状CSV报告时出错: {e}")
            return None
    
    def get_depression_statistics(self) -> Dict:
        """获取抑郁症状数据统计信息"""
        try:
            depression_file = os.path.join(self.data_dir, 'depression_records.csv')
            if not os.path.exists(depression_file):
                return {}
            
            df = pd.read_csv(depression_file)
            
            if df.empty:
                return {}
            
            stats = {
                'total_users': df['user_id'].nunique(),
                'total_sessions': len(df),
                'today_sessions': len(df[df['timestamp'].str.startswith(datetime.now().strftime('%Y-%m-%d'))]),
                'risk_distribution': df['risk_level'].value_counts().to_dict(),
                'average_age': df['age'].mean(),
                'average_work_hours': df['weekly_hours'].mean(),
                'most_common_risk': df['risk_level'].mode().iloc[0] if not df['risk_level'].mode().empty else '未知'
            }
            
            return stats
            
        except Exception as e:
            st.error(f"获取抑郁症状统计信息时出错: {e}")
            return {}


def create_data_management_page():
    """创建数据管理页面"""
    st.title("数据管理与报告导出")
    data_manager = DataManager()
    
    # 获取统计信息
    stats = data_manager.get_user_statistics()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总用户数", stats['total_users'])
        
        with col2:
            st.metric("总评估次数", stats['total_sessions'])
        
        with col3:
            st.metric("今日评估", stats['today_sessions'])
        
        with col4:
            st.metric("最常见风险", stats.get('most_common_risk', '未知'))
        
        # 风险分布图表
        st.subheader("风险等级分布")
        if stats['risk_distribution']:
            risk_df = pd.DataFrame(list(stats['risk_distribution'].items()), 
                                 columns=['风险等级', '数量'])
            order = ['低风险', '中风险', '高风险', '未知']
            risk_df['排序'] = risk_df['风险等级'].apply(lambda x: order.index(x) if x in order else 99)
            risk_df = risk_df.sort_values('排序').drop(columns=['排序'])
            total_count = risk_df['数量'].sum()
            risk_df['占比'] = risk_df['数量'] / total_count

            color_map = {'低风险': '#2E7D32', '中风险': '#F9A825', '高风险': '#C62828', '未知': '#6F42C1'}
            fig = px.bar(
                risk_df,
                x='风险等级',
                y='数量',
                color='风险等级',
                title='风险等级分布总览',
                text=risk_df['占比'].map(lambda x: f'{x:.1%}'),
                color_discrete_map=color_map
            )
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='rgba(248,249,252,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13),
                margin=dict(l=20, r=20, t=56, b=20),
                title=dict(x=0.02, xanchor='left'),
                showlegend=False,
                xaxis_title='风险等级',
                yaxis_title='人数',
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)')
            )
            fig.update_traces(
                marker_line_color='#FFFFFF',
                marker_line_width=1.2,
                textposition='outside',
                hovertemplate='风险等级: %{x}<br>人数: %{y}<extra></extra>'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无用户数据")
    
    # 数据导出功能
    st.subheader("数据导出")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("导出完整用户数据(CSV)"):
            try:
                if os.path.exists(data_manager.user_data_file):
                    with open(data_manager.user_data_file, 'rb') as f:
                        st.download_button(
                            label="下载用户数据CSV",
                            data=f.read(),
                            file_name=f"user_data_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key="download_user_data"
                        )
                else:
                    st.warning("暂无用户数据可导出")
            except Exception as e:
                st.error(f"导出数据时出错: {e}")
    
    with col2:
        if st.button("清理旧报告文件(30天前)"):
            data_manager.cleanup_old_reports()
            st.success("旧报告文件清理完成")
    
    st.subheader("最近评估记录")
    try:
        if os.path.exists(data_manager.user_data_file):
            df = pd.read_csv(data_manager.user_data_file)
            if not df.empty:
                cols = [c for c in ['timestamp', 'prediction_type', 'risk_level', 'risk_probability', 'confidence'] if c in df.columns]
                recent_records = df.tail(5)[cols]
                st.dataframe(recent_records)
            else:
                st.info("暂无评估记录")
        else:
            st.info("暂无评估记录")
    except Exception as e:
        st.error(f"加载记录时出错: {e}")

if __name__ == "__main__":
    dm = DataManager()
    
    test_input = {
        'age': 30, 'work_years': 5, 'position_years': 2, 'income': '中',
        'weekly_hours': 40, 'alcohol': 0, 'mid_exercise': 2, 'high_exercise': 1,
        'life_satisfaction': 6, 'fatigue_level': 1, 'education': '本科', 
        'shift_work': '否', 'night_shift': '否', 'sleep_disorder': '否'
    }
    
    test_result = {
        'prediction': 0, 'probability': 0.45, 'risk_level': '中风险', 'confidence': 0.85
    }
    
    session_id = dm.save_user_record(test_input, test_result)
    print(f"测试记录保存成功，会话ID: {session_id}")
    
    stats = dm.get_user_statistics()
    print("统计信息:", stats)