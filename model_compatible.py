import pandas as pd
import numpy as np
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import os

class CompatibleOccupationalStressModel:
    """兼容版本的职业紧张模型，避免版本冲突"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        # 以“职业紧张”新特征列为准（要求数据集中列名完全匹配）
        self.feature_names = [
            '工龄',
            '日均工作时长',
            '工作休息时长',
            '付出与认可',
            '轮班制承受度',
            '加班情况',
            '工作自由度',
            '做事兴趣',
            '自伤想法',
            '情绪影响',
            '疲劳蓄积等级',
        ]
        self.target_name = '是否职业紧张'
        self.model_path = 'models/compatible_stress_model.pkl'
        self.scaler_path = 'models/compatible_scaler.pkl'
        self.data_path = '9066新业态数据-特征工程后.xlsx'
        
    def get_all_features(self):
        """获取所有可用的特征列表"""
        return self.feature_names
        
    def load_data(self, excel_path=None, target_name=None):
        """加载和预处理数据"""
        if excel_path is None:
            excel_path = self.data_path
        if target_name is None:
            target_name = self.target_name
            
        # 使用绝对路径确保文件正确加载
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_excel_path = os.path.join(base_dir, excel_path)
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 读取数据（使用绝对路径）
        df = pd.read_excel(full_excel_path)
        
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 创建原始特征名称的副本用于训练
        original_feature_names = self.feature_names.copy()
        
        # 验证特征列是否存在（新特征缺失则直接报错，避免训练出“错误特征集”的模型）
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"数据集中缺少职业紧张特征列: {missing_features}。"
                f"请确认 Excel 列名与 feature_names 完全一致。"
            )
                
        # 分离特征和目标变量
        X = df[self.feature_names]
        y = df[target_name].astype(int)
        
        # 保存实际使用的特征名称到训练特征文件
        with open(self.model_path.replace('.pkl', '_features.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # 恢复原始特征名称用于预测时的输入处理
        self.feature_names = original_feature_names
        
        # 处理类别不平衡（手动实现重采样）
        class_0_count = (y == 0).sum()
        class_1_count = (y == 1).sum()
        
        if class_1_count < class_0_count:
            # 对少数类进行上采样
            minority_indices = y[y == 1].index
            oversampled_indices = np.random.choice(
                minority_indices, 
                size=class_0_count - class_1_count, 
                replace=True
            )
            X_oversampled = pd.concat([X, X.loc[oversampled_indices]])
            y_oversampled = pd.concat([y, y.loc[oversampled_indices]])
            X, y = X_oversampled, y_oversampled
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 保存标准化器
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        return X_scaled, y
    
    def optimize_hyperparameters(self, X, y):
        """手动实现超参数优化（CatBoost）"""
        best_score = 0
        best_params = {}
        
        # 简单的参数搜索
        param_combinations = [
            {'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
            {'iterations': 500, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5},
            {'iterations': 800, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 7},
            {'iterations': 500, 'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
        ]
        
        for params in param_combinations:
            # 5折交叉验证
            scores = []
            for _ in range(5):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=np.random.randint(1000)
                )
                
                model = CatBoostClassifier(
                    iterations=params['iterations'],
                    depth=params['depth'],
                    learning_rate=params['learning_rate'],
                    l2_leaf_reg=params['l2_leaf_reg'],
                    loss_function='Logloss',
                    eval_metric='F1',
                    auto_class_weights='Balanced',
                    random_seed=42,
                    verbose=False
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        print(f"最佳F1分数: {best_score:.3f}")
        print(f"最佳参数: {best_params}")
        
        # 使用最佳参数训练最终模型
        final_model = CatBoostClassifier(
            iterations=best_params['iterations'],
            depth=best_params['depth'],
            learning_rate=best_params['learning_rate'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            loss_function='Logloss',
            eval_metric='F1',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
        
        return final_model
    
    def train_model(self, excel_path=None):
        """训练兼容版本的模型"""
        print("🚀 开始训练兼容模型...")
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 加载数据
        X, y = self.load_data(excel_path)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 优化超参数
        self.model = self.optimize_hyperparameters(X_train, y_train)
        
        # 训练最终模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"测试集准确率: {accuracy:.3f}")
        print(f"测试集F1分数: {f1:.3f}")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ 模型已保存到: {self.model_path}")
        return accuracy
    
    def predict_stress(self, input_data):
        """预测职业紧张风险"""
        if self.model is None:
            self.load_model()
        
        if isinstance(input_data, dict):
            # 加载训练时实际使用的特征名称
            features_path = self.model_path.replace('.pkl', '_features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    actual_feature_names = pickle.load(f)
            else:
                actual_feature_names = self.feature_names
            
            input_df = pd.DataFrame([input_data])
            # 只保留训练时使用的特征
            missing = [c for c in actual_feature_names if c not in input_df.columns]
            if missing:
                raise ValueError(
                    f"输入数据缺少模型所需特征列: {missing}。"
                    f"如果你刚刚替换了特征集，请删除旧模型文件（models/*.pkl 及 *_features.pkl）并用新数据重新训练。"
                )
            input_df = input_df[actual_feature_names]
            
            # 使用保存的标准化器
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
        else:
            input_scaled = input_data
        
        probability = self.model.predict_proba(input_scaled)[0]
        prediction = self.model.predict(input_scaled)[0]
        
        risk_prob = probability[1]
        if risk_prob < 0.3:
            risk_level = "低风险"
        elif risk_prob < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
            
        return {
            'prediction': int(prediction),
            'probability': risk_prob,
            'risk_level': risk_level,
            'confidence': max(probability)
        }
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("⚠️ 模型文件不存在，请先训练模型")
            self.train_model()
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            self.load_model()
        
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
        else:
            importance = self.model.feature_importances_
        return sorted(zip(self.feature_names, importance), 
                     key=lambda x: x[1], reverse=True)

def process_user_input(user_input):
    """处理用户输入，转换为模型需要的格式"""
    # 兼容保留：当前 Streamlit 已直接按新列名传入 dict，这里仅做兜底/校验型转换
    return {
        '工龄': user_input.get('工龄', user_input.get('work_years', 0)),
        '日均工作时长': user_input.get('日均工作时长', user_input.get('daily_work_hours', 0)),
        '工作休息时长': user_input.get('工作休息时长', user_input.get('daily_rest_hours', 0)),
        '付出与认可': user_input.get('付出与认可', 0),
        '轮班制承受度': user_input.get('轮班制承受度', 0),
        '加班情况': user_input.get('加班情况', 0),
        '工作自由度': user_input.get('工作自由度', 0),
        '做事兴趣': user_input.get('做事兴趣', 0),
        '自伤想法': user_input.get('自伤想法', 0),
        '情绪影响': user_input.get('情绪影响', 0),
        '疲劳蓄积等级': user_input.get('疲劳蓄积等级', user_input.get('疲劳程度分级', 0)),
    }

class CompatibleDepressionModel:
    """兼容版本的抑郁症状预测模型"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        # 以“抑郁症状”新特征列为准（要求数据集中列名完全匹配）
        self.feature_names = [
            '本地居住时间',
            '劳动确定方式',
            '工龄',
            '日均工作时长',
            '工作休息时长',
            '夜班次数',
            '付出与认可',
            '轮班制承受度',
            '加班情况',
            '工作自由度',
            '做事兴趣',
            '自伤想法',
            '情绪影响',
            '吸烟量',
            '油炸摄入频率',
            '甜点摄入频率',
            '入睡时间',
            '难入睡',
            '醒太早',
            '健康状况',
            '因病休工',
            '疲劳蓄积等级',
        ]
        self.target_name = '是否抑郁症状'
        self.model_path = 'models/compatible_depression_model.pkl'
        self.scaler_path = 'models/compatible_depression_scaler.pkl'
        self.data_path = '9066新业态数据-特征工程后.xlsx'
        
    def get_all_features(self):
        """获取所有可用的特征列表"""
        return self.feature_names
        
    def load_data(self, excel_path=None, target_name=None):
        """加载和预处理数据"""
        if excel_path is None:
            excel_path = self.data_path
        if target_name is None:
            target_name = self.target_name
            
        # 使用绝对路径确保文件正确加载
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_excel_path = os.path.join(base_dir, excel_path)
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 读取数据（使用绝对路径）
        df = pd.read_excel(full_excel_path)
        
        # 处理缺失值
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # 创建原始特征名称的副本用于训练
        original_feature_names = self.feature_names.copy()
        
        # 验证特征列是否存在（新特征缺失则直接报错，避免训练出“错误特征集”的模型）
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"数据集中缺少抑郁症状特征列: {missing_features}。"
                f"请确认 Excel 列名与 feature_names 完全一致。"
            )
                
        # 分离特征和目标变量
        X = df[self.feature_names]
        y = df[target_name].astype(int)
        
        # 保存实际使用的特征名称到训练特征文件
        with open(self.model_path.replace('.pkl', '_features.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # 恢复原始特征名称用于预测时的输入处理
        self.feature_names = original_feature_names
        
        # 处理类别不平衡（手动实现重采样）
        class_0_count = (y == 0).sum()
        class_1_count = (y == 1).sum()
        
        if class_1_count < class_0_count:
            # 对少数类进行上采样
            minority_indices = y[y == 1].index
            oversampled_indices = np.random.choice(
                minority_indices, 
                size=class_0_count - class_1_count, 
                replace=True
            )
            X_oversampled = pd.concat([X, X.loc[oversampled_indices]])
            y_oversampled = pd.concat([y, y.loc[oversampled_indices]])
            X, y = X_oversampled, y_oversampled
        
        # 数据标准化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # 保存标准化器
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        return X_scaled, y
    
    def optimize_hyperparameters(self, X, y):
        """手动实现超参数优化（CatBoost）"""
        best_score = 0
        best_params = {}
        
        # 简单的参数搜索
        param_combinations = [
            {'iterations': 300, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
            {'iterations': 500, 'depth': 8, 'learning_rate': 0.03, 'l2_leaf_reg': 5},
            {'iterations': 800, 'depth': 6, 'learning_rate': 0.03, 'l2_leaf_reg': 7},
            {'iterations': 500, 'depth': 10, 'learning_rate': 0.05, 'l2_leaf_reg': 3},
        ]
        
        for params in param_combinations:
            # 5折交叉验证
            scores = []
            for _ in range(5):
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=np.random.randint(1000)
                )
                
                model = CatBoostClassifier(
                    iterations=params['iterations'],
                    depth=params['depth'],
                    learning_rate=params['learning_rate'],
                    l2_leaf_reg=params['l2_leaf_reg'],
                    loss_function='Logloss',
                    eval_metric='F1',
                    auto_class_weights='Balanced',
                    random_seed=42,
                    verbose=False
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
        
        print(f"最佳F1分数: {best_score:.3f}")
        print(f"最佳参数: {best_params}")
        
        # 使用最佳参数训练最终模型
        final_model = CatBoostClassifier(
            iterations=best_params['iterations'],
            depth=best_params['depth'],
            learning_rate=best_params['learning_rate'],
            l2_leaf_reg=best_params['l2_leaf_reg'],
            loss_function='Logloss',
            eval_metric='F1',
            auto_class_weights='Balanced',
            random_seed=42,
            verbose=False
        )
        
        return final_model
    
    def train_model(self, excel_path=None):
        """训练兼容版本的抑郁症状模型"""
        print("?? 开始训练抑郁症状预测模型...")
        
        # 确保models目录存在
        os.makedirs('models', exist_ok=True)
        
        # 加载数据
        X, y = self.load_data(excel_path)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 优化超参数
        self.model = self.optimize_hyperparameters(X_train, y_train)
        
        # 训练最终模型
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"测试集准确率: {accuracy:.3f}")
        print(f"测试集F1分数: {f1:.3f}")
        print(classification_report(y_test, y_pred))
        
        # 保存模型
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"✅ 抑郁症状模型已保存到: {self.model_path}")
        return accuracy
    
    def predict_depression(self, input_data):
        """预测抑郁症状风险"""
        if self.model is None:
            self.load_model()
        
        if isinstance(input_data, dict):
            # 加载训练时实际使用的特征名称
            features_path = self.model_path.replace('.pkl', '_features.pkl')
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    actual_feature_names = pickle.load(f)
            else:
                actual_feature_names = self.feature_names
            
            input_df = pd.DataFrame([input_data])
            # 只保留训练时使用的特征
            missing = [c for c in actual_feature_names if c not in input_df.columns]
            if missing:
                raise ValueError(
                    f"输入数据缺少模型所需特征列: {missing}。"
                    f"如果你刚刚替换了特征集，请删除旧模型文件（models/*.pkl 及 *_features.pkl）并用新数据重新训练。"
                )
            input_df = input_df[actual_feature_names]
            
            # 使用保存的标准化器
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                input_scaled = scaler.transform(input_df)
            else:
                input_scaled = input_df.values
        else:
            input_scaled = input_data
        
        probability = self.model.predict_proba(input_scaled)[0]
        prediction = self.model.predict(input_scaled)[0]
        
        risk_prob = probability[1]
        if risk_prob < 0.3:
            risk_level = "低风险"
        elif risk_prob < 0.7:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
            
        return {
            'prediction': int(prediction),
            'probability': risk_prob,
            'risk_level': risk_level,
            'confidence': max(probability)
        }
    
    def load_model(self):
        """加载模型"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("⚠️ 抑郁症状模型文件不存在，请先训练模型")
            self.train_model()
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.model is None:
            self.load_model()
        
        if hasattr(self.model, 'get_feature_importance'):
            importance = self.model.get_feature_importance()
        else:
            importance = self.model.feature_importances_
        return sorted(zip(self.feature_names, importance), 
                     key=lambda x: x[1], reverse=True)


def process_depression_input(user_input):
    """处理用户输入，转换为抑郁症状模型需要的格式"""
    # 兼容保留：当前 Streamlit 已直接按新列名传入 dict，这里仅做兜底/校验型转换
    return {
        '本地居住时间': user_input.get('本地居住时间', 0),
        '劳动确定方式': user_input.get('劳动确定方式', 0),
        '工龄': user_input.get('工龄', user_input.get('work_years', 0)),
        '日均工作时长': user_input.get('日均工作时长', user_input.get('daily_work_hours', 0)),
        '工作休息时长': user_input.get('工作休息时长', user_input.get('daily_rest_hours', 0)),
        '夜班次数': user_input.get('夜班次数', 0),
        '付出与认可': user_input.get('付出与认可', 0),
        '轮班制承受度': user_input.get('轮班制承受度', 0),
        '加班情况': user_input.get('加班情况', 0),
        '工作自由度': user_input.get('工作自由度', 0),
        '做事兴趣': user_input.get('做事兴趣', 0),
        '自伤想法': user_input.get('自伤想法', 0),
        '情绪影响': user_input.get('情绪影响', 0),
        '吸烟量': user_input.get('吸烟量', user_input.get('smoking_amount', 0)),
        '油炸摄入频率': user_input.get('油炸摄入频率', 0),
        '甜点摄入频率': user_input.get('甜点摄入频率', 0),
        '入睡时间': user_input.get('入睡时间', 0),
        '难入睡': user_input.get('难入睡', 0),
        '醒太早': user_input.get('醒太早', 0),
        '健康状况': user_input.get('健康状况', 0),
        '因病休工': user_input.get('因病休工', 0),
        '疲劳蓄积等级': user_input.get('疲劳蓄积等级', user_input.get('疲劳程度分级', 0)),
    }


# 训练职业紧张模型
if __name__ == "__main__":
    print("🎯 职业紧张模型兼容版本")
    print("=" * 50)
    
    model = CompatibleOccupationalStressModel()
    accuracy = model.train_model()
    
    print("=" * 50)
    print(f"✅ 职业紧张模型训练完成！准确率: {accuracy:.3f}")
    
    print("\n" + "=" * 50)
    print("🎯 抑郁症状预测模型")
    print("=" * 50)
    
    depression_model = CompatibleDepressionModel()
    depression_accuracy = depression_model.train_model()
    
    print("=" * 50)
    print(f"✅ 抑郁症状模型训练完成！准确率: {depression_accuracy:.3f}")