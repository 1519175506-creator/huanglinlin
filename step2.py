"""
航运保险货物描述 - 文本预处理脚本
模块（二）2.2：文本预处理
功能：分词、去噪、标准化、特征提取
"""
import pandas as pd
import numpy as np
import re
import jieba
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("航运保险货物描述 - 文本预处理")
print("=" * 70)

# ==================== 1. 数据加载 ====================
print("\n[1/6] 加载数据...")
df = pd.read_excel("/Users/dori/Documents/Fintech/保险project/货物描述信息2025.xlsx")

# 统一列名
if '货物描述' in df.columns:
    df.rename(columns={'货物描述': 'raw_text'}, inplace=True)
elif 'itemDescription' in df.columns:
    df.rename(columns={'itemDescription': 'raw_text'}, inplace=True)
else:
    df.rename(columns={df.columns[0]: 'raw_text'}, inplace=True)

print(f"  原始数据: {len(df)} 条")

# ==================== 2. 行业词典加载 ====================
print("\n[2/6] 加载行业词典...")

# 航运/贸易行业术语
industry_terms = [
    # 货物形态
    '散装', '整箱', '拼箱', '托盘', '捆扎', '卷装', '袋装', '桶装',
    # 货物状态
    '冷冻', '冷藏', '常温', '恒温', '干燥', '鲜活', '冰鲜',
    # 材料类
    '不锈钢', '碳钢', '合金钢', '铝合金', '铜合金', '钛合金',
    '碳纤维', '玻璃纤维', '聚氨酯', '聚乙烯', '聚丙烯', '聚酯',
    '陶瓷', '石墨', '石英', '硅胶', '橡胶',
    # 产品类
    '零部件', '配件', '散件', '半成品', '原料', '成品',
    '电动滑板车', '太阳能板', '锂电池', '发电机', '电动机',
    '阀门', '轴承', '法兰', '管件', '紧固件',
    # 化工类
    '提取物', '萃取物', '粉末', '颗粒', '液体', '浆料',
    # 纺织类
    '针织', '梭织', '牛仔', '休闲裤', 'T恤', '衬衫',
    # 设备类
    '激光切割机', '注塑机', '机床', '机器人', '传感器',
    # 包装类
    '纸箱', '木箱', '编织袋', '集装袋',
    # 风险相关
    '易燃', '易爆', '腐蚀性', '有毒', '精密仪器',
    # 除外标的
    '烟花爆竹', '活动物', '鲜活货', '艺术品', '古董', '珠宝',
]

for term in industry_terms:
    jieba.add_word(term)
    jieba.add_word(term.upper())  # 同时添加大写版本

print(f"  已加载 {len(industry_terms)} 个行业术语")

# ==================== 3. 停用词定义 ====================
print("\n[3/6] 定义停用词表...")

# 中文停用词
stopwords_cn = {
    '的', '了', '在', '是', '和', '及', '与', '或', '等', '为', '以',
    '其', '这', '那', '该', '此', '本', '各', '之', '也', '就', '都',
    '而', '且', '但', '却', '则', '于', '把', '被', '让', '从', '对',
    '向', '到', '在', '当', '所', '个', '种', '些', '每', '某', '另',
    '已', '还', '又', '再', '更', '最', '只', '仅', '可', '能', '会',
    '要', '将', '应', '可以', '需要', '必须', '能够', '可能',
    '名称', '规格', '型号', '数量', '单位', '备注',  # 表格标题残留
}

# 英文停用词
stopwords_en = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'including', 'such', 'this', 'that', 'these', 'those', 'it', 'its',
    'and', 'or', 'but', 'not', 'no', 'if', 'then', 'than', 'too', 'very',
    'just', 'also', 'now', 'here', 'there', 'when', 'where', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
    'only', 'own', 'same', 'so', 'up', 'out', 'about', 'over', 'under',
    'again', 'further', 'once',
    # 物流停用词
    'hs', 'code', 'cif', 'fob', 'per', 'invoice', 'packing', 'list',
    'total', 'net', 'gross', 'weight', 'volume', 'quantity',
}

print(f"  中文停用词: {len(stopwords_cn)} 个")
print(f"  英文停用词: {len(stopwords_en)} 个")

# ==================== 4. 同义词映射表 ====================
print("\n[4/6] 构建同义词映射表...")

synonym_map = {
    # 零部件类 -> 统一为 "PARTS"
    '零件': 'PARTS', '部件': 'PARTS', '配件': 'PARTS', '组件': 'PARTS',
    'PARTS': 'PARTS', 'PART': 'PARTS', 'COMPONENT': 'PARTS',
    'COMPONENTS': 'PARTS', 'ACCESSORIES': 'PARTS', 'SPARE': 'PARTS',
    
    # 机械设备类 -> 统一为 "MACHINE"
    '机械': 'MACHINE', '机器': 'MACHINE', '设备': 'MACHINE',
    'MACHINE': 'MACHINE', 'MACHINERY': 'MACHINE',
    'EQUIPMENT': 'MACHINE', 'EQUIP': 'MACHINE',
    
    # 钢铁类 -> 统一为 "STEEL"
    '钢': 'STEEL', '钢铁': 'STEEL', '钢材': 'STEEL',
    'STEEL': 'STEEL', 'IRON': 'STEEL',
    
    # 铝类 -> 统一为 "ALUMINUM"
    '铝': 'ALUMINUM', '铝合金': 'ALUMINUM',
    'ALUMINUM': 'ALUMINUM', 'ALUMINIUM': 'ALUMINUM',
    
    # 塑料类 -> 统一为 "PLASTIC"
    '塑料': 'PLASTIC', '塑胶': 'PLASTIC',
    'PLASTIC': 'PLASTIC', 'PLASTICS': 'PLASTIC',
    
    # 纺织服装类 -> 保留区分度
    '服装': 'GARMENT', '衣服': 'GARMENT', '成衣': 'GARMENT',
    'GARMENT': 'GARMENT', 'GARMENTS': 'GARMENT',
    'APPAREL': 'GARMENT', 'CLOTHING': 'GARMENT',
    
    # 面料类 -> 统一为 "FABRIC"
    '面料': 'FABRIC', '布料': 'FABRIC', '织物': 'FABRIC',
    'FABRIC': 'FABRIC', 'TEXTILE': 'FABRIC',
    
    # 化工提取物类
    '提取物': 'EXTRACT', '萃取物': 'EXTRACT',
    'EXTRACT': 'EXTRACT', 'EXTRACTS': 'EXTRACT',
    
    # 粉末类
    '粉末': 'POWDER', '粉': 'POWDER',
    'POWDER': 'POWDER',
    
    # 液体类
    '液体': 'LIQUID', '液': 'LIQUID',
    'LIQUID': 'LIQUID',
    
    # 包装类 -> 保留，帮助推断货物形态
    '纸箱': 'CARTON', '木箱': 'WOODEN_BOX',
    '编织袋': 'WOVEN_BAG', '集装袋': 'BULK_BAG',
    
    # 其他常见同义词
    '马达': 'MOTOR', '电动机': 'MOTOR', '电机': 'MOTOR',
    'MOTOR': 'MOTOR', 'ENGINE': 'MOTOR',
    '管子': 'PIPE', '管材': 'PIPE', '管道': 'PIPE',
    'PIPE': 'PIPE', 'TUBE': 'PIPE', 'TUBING': 'PIPE',
    '板': 'PLATE', '板材': 'PLATE',
    'PLATE': 'PLATE', 'SHEET': 'PLATE', 'PANEL': 'PLATE',
    '阀': 'VALVE', '阀门': 'VALVE',
    'VALVE': 'VALVE',
    '泵': 'PUMP',
    'PUMP': 'PUMP',
}

print(f"  同义词映射: {len(synonym_map)} 条")

# ==================== 5. 核心预处理函数 ====================
print("\n[5/6] 执行预处理...")

def extract_hs_code(text):
    """步骤1: 提取HS CODE（清洗前先行提取）"""
    text_upper = str(text).upper()
    patterns = [
        r'HS\s*CODE:?\s*(\d{4,12})',
        r'HS\s*CODE\s*NO:?\s*(\d{4,12})',
        r'HS\s*NO:?\s*(\d{4,12})',
        r'HS:?\s*(\d{4,12})',
        r'\bHS\s*(\d{4,12})\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            code = match.group(1)
            if len(code) >= 6:
                return code[:10]  # 保留前10位
    return None

def remove_logistics_noise(text):
    """步骤2: 去除物流相关噪声"""
    text = str(text)
    
    # 集装箱号 (4个字母+7位数字)
    text = re.sub(r'\b[A-Z]{4}\d{7}\b', ' ', text)
    
    # 物流参考号
    text = re.sub(r'(?:CNDC|ML-CN|S/|B/L)\s*\w*\d+', ' ', text, flags=re.IGNORECASE)
    
    # 包装声明
    text = re.sub(r'\d{1,4}\s*(?:PACKAGES?|PKGS?|CTNS?|CARTONS?)\s*(?:=)?\s*\d*\s*(?:PALLETS?)?', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,4}\s*(?:PALLETS?|ROLLS?|BUNDLES?|DRUMS?|BOXES?|BAGS?)', ' ', text, flags=re.IGNORECASE)
    
    # 重量体积
    text = re.sub(r'\d{1,6}\.?\d*\s*(?:KGS?|KG|LBS?|G|TONS?)\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d{1,6}\.?\d*\s*(?:CBM|CUBIC|M3|LITERS?|LTRS?)\b', ' ', text, flags=re.IGNORECASE)
    
    # VIN码
    text = re.sub(r'\b[A-HJ-NPR-Z0-9]{17}\b', ' ', text)
    text = re.sub(r'VIN\s*[:\s]*[\w\d]+', ' ', text, flags=re.IGNORECASE)
    
    # HS CODE声明
    text = re.sub(r'HS\s*CODE:?\s*\d{4,12}', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'HS\s*NO:?\s*\d{4,12}', ' ', text, flags=re.IGNORECASE)
    
    # 金额
    text = re.sub(r'(?:USD|EUR|RMB|CNY|TOTAL|AMOUNT)\s*[\d,\.]+', ' ', text, flags=re.IGNORECASE)
    
    # 邮箱/电话/网址
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', ' ', text)
    text = re.sub(r'(?:TEL|FAX|PHONE)[\s:]*[\d\-\(\)\+]+', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'https?://\S+', ' ', text)
    
    # 物流行标识
    text = re.sub(r'AS\s+PER\s+(?:INVOICE|BL|B/L|ORDER)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'CIF\s+\w+\s+(?:SEAPORT|PORT)', ' ', text, flags=re.IGNORECASE)
    
    return text

def clean_special_chars(text):
    """步骤3: 清理特殊字符"""
    text = str(text)
    
    # 保留有用字符：字母、数字、中文、常用标点
    # 去除其他特殊符号
    text = re.sub(r'[^\u4e00-\u9fff\w\s\-\.\,\;\:\/\(\)\[\]\{\}\#\+\=\&\@\%]', ' ', text)
    
    # 合并多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def normalize_tokens(text, synonym_dict):
    """步骤4: 同义词标准化"""
    words = str(text).upper().split()
    normalized = []
    for word in words:
        normalized.append(synonym_dict.get(word, word))
    return ' '.join(normalized)

def tokenize_mixed_text(text):
    """步骤5: 中英文混合分词"""
    text = str(text).strip()
    if not text:
        return [], []
    
    # 分离中文和英文
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
    chinese_text = ' '.join(chinese_chars)
    
    # 英文部分：去掉中文后的剩余
    english_text = re.sub(r'[\u4e00-\u9fff]+', ' ', text)
    english_words = re.findall(r'\b[a-zA-Z]{2,}\b', english_text.upper())
    
    # 中文分词
    chinese_tokens = []
    if chinese_text:
        for word in jieba.cut(chinese_text):
            word = word.strip()
            if len(word) > 1 and word not in stopwords_cn:
                chinese_tokens.append(word)
    
    # 英文过滤
    english_tokens = [w for w in english_words if w not in stopwords_en and len(w) > 1]
    
    return chinese_tokens, english_tokens

def preprocess_pipeline(text):
    """完整预处理管道"""
    if pd.isna(text) or str(text).strip() == '':
        return {
            'hs_code': None,
            'clean_text': '',
            'cn_tokens': [],
            'en_tokens': [],
            'all_tokens': [],
            'char_count': 0,
            'token_count': 0
        }
    
    # Step 1: 提取HS CODE
    hs_code = extract_hs_code(text)
    
    # Step 2: 去除物流噪声
    clean = remove_logistics_noise(text)
    
    # Step 3: 清洗特殊字符
    clean = clean_special_chars(clean)
    
    # Step 4: 标准化同义词
    clean = normalize_tokens(clean, synonym_map)
    
    # Step 5: 分词
    cn_tokens, en_tokens = tokenize_mixed_text(clean)
    
    all_tokens = cn_tokens + en_tokens
    
    return {
        'hs_code': hs_code,
        'clean_text': clean,
        'cn_tokens': cn_tokens,
        'en_tokens': en_tokens,
        'all_tokens': all_tokens,
        'char_count': len(clean),
        'token_count': len(all_tokens)
    }

# 批量处理
results = []
for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"  处理进度: {idx}/{len(df)}")
    result = preprocess_pipeline(row['raw_text'])
    results.append(result)

# 合并结果
df['hs_code'] = [r['hs_code'] for r in results]
df['clean_text'] = [r['clean_text'] for r in results]
df['cn_tokens'] = [r['cn_tokens'] for r in results]
df['en_tokens'] = [r['en_tokens'] for r in results]
df['all_tokens'] = [r['all_tokens'] for r in results]
df['char_count_clean'] = [r['char_count'] for r in results]
df['token_count'] = [r['token_count'] for r in results]

print(f"  预处理完成！")

# ==================== 6. 结果统计与输出 ====================
print("\n[6/6] 预处理结果统计...")

# 基本统计
valid_count = df[df['clean_text'] != ''].shape[0]
empty_count = df[df['clean_text'] == ''].shape[0]
hs_extracted = df['hs_code'].notna().sum()

print(f"\n  {'='*50}")
print(f"  预处理结果摘要")
print(f"  {'='*50}")
print(f"  总样本数:           {len(df)}")
print(f"  有效样本（非空）:   {valid_count} ({valid_count/len(df)*100:.1f}%)")
print(f"  空样本（清洗后）:   {empty_count} ({empty_count/len(df)*100:.1f}%)")
print(f"  提取到HS CODE:      {hs_extracted} ({hs_extracted/len(df)*100:.1f}%)")
print(f"  清洗后平均字符数:   {df['char_count_clean'].mean():.1f}")
print(f"  清洗后中位数字符数: {df['char_count_clean'].median():.1f}")
print(f"  平均Token数:        {df['token_count'].mean():.1f}")
print(f"  中位数Token数:      {df['token_count'].median():.1f}")

# 预处理前后对比样例
print(f"\n  {'='*50}")
print(f"  预处理前后对比样例")
print(f"  {'='*50}")

samples = [
    "WORKED SLATE\nONEU2133570/CNDC19297/20GP/64PACKAGES/26550.000KGS/20.000CBM",
    "FIBER LASER CUTTING MACHINE\nMODEL: D-POWER 2560FCCD 6000W RAYCUS\nMANUROBE",
    "Lady's knitted T-shirt 95% COTTON 5%SPANDEX\nWOMEN'S KNITTED PULLOVER 78%RA",
    "水彩颜料、画笔、绘画本等及配件",
    "电动滑板车(无电池)\nScooter YGW 5.0",
    "HYBRID CAR\nVIN:LFMICU1BR6S0140461\nLFMICU1BR0S0140472",
    "MacTex W2 40.05 36 Roll (Roll Size: 5.2 x 200m)\nHS CODE: 5407.61.90",
]

for i, sample in enumerate(samples, 1):
    result = preprocess_pipeline(sample)
    print(f"\n  【样例{i}】")
    print(f"  原始: {sample[:100]}...")
    print(f"  清洗: {result['clean_text'][:100]}")
    print(f"  分词: {result['all_tokens'][:20]}")
    print(f"  HS:   {result['hs_code']}")

# 保存预处理结果
output_path = "/Users/dori/Documents/Fintech/保险project/货物描述_预处理结果.xlsx"
df_export = df[['raw_text', 'clean_text', 'hs_code', 'all_tokens', 'cn_tokens', 'en_tokens', 'char_count_clean', 'token_count']].copy()
df_export['all_tokens'] = df_export['all_tokens'].apply(lambda x: ' '.join(x) if x else '')
df_export['cn_tokens'] = df_export['cn_tokens'].apply(lambda x: ' '.join(x) if x else '')
df_export['en_tokens'] = df_export['en_tokens'].apply(lambda x: ' '.join(x) if x else '')
df_export.to_excel(output_path, index=False)
print(f"\n  ✅ 预处理结果已保存至: {output_path}")

print("\n" + "=" * 70)
print("文本预处理完成！")
print("=" * 70)
