import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
from openai import AsyncOpenAI
from config.app_config import (
    ONE_API_KEY,
    ONE_API_BASE_URL,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME
)

async def test_llm_connection():
    """测试LLM连接"""
    print("=" * 60)
    print("LLM连接测试")
    print("=" * 60)
    print(f"API Base URL: {ONE_API_BASE_URL}")
    print(f"API Key: {ONE_API_KEY[:10]}...")
    print(f"Model: {LLM_MODEL_NAME}")
    print("=" * 60)
    
    client = AsyncOpenAI(
        api_key=ONE_API_KEY,
        base_url=ONE_API_BASE_URL
    )
    
    try:
        print("\n正在测试LLM连接...")
        response = await client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": "你好，请用一句话介绍自己"}],
            temperature=0.1,
            max_tokens=100
        )
        result = response.choices[0].message.content.strip()
        print(f"✓ LLM连接成功")
        print(f"响应: {result}")
        return True
    except Exception as e:
        print(f"✗ LLM连接失败: {e}")
        return False

async def test_embedding_connection():
    """测试Embedding连接"""
    print("\n" + "=" * 60)
    print("Embedding连接测试")
    print("=" * 60)
    print(f"API Base URL: {ONE_API_BASE_URL}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    print("=" * 60)
    
    client = AsyncOpenAI(
        api_key=ONE_API_KEY,
        base_url=ONE_API_BASE_URL
    )
    
    try:
        print("\n正在测试Embedding连接...")
        response = await client.embeddings.create(
            input="测试文本",
            model=EMBEDDING_MODEL_NAME
        )
        vector = response.data[0].embedding
        print(f"✓ Embedding连接成功")
        print(f"向量维度: {len(vector)}")
        return True
    except Exception as e:
        print(f"✗ Embedding连接失败: {e}")
        return False

async def test_feature_extractor():
    """测试FeatureExtractor"""
    print("\n" + "=" * 60)
    print("FeatureExtractor测试")
    print("=" * 60)
    
    from api.core.feature_extractor import g_feature_extractor
    
    # test_text = "人工智能在医疗影像诊断中的应用研究"
    test_text = """
    美沙拉嗪灌肠液（化学药品注册分类 3）适应症
成人活动期轻度至中度远端溃疡性结肠炎、直肠乙状结肠炎或直肠炎
项目简介
溃疡性结肠炎是一种发病机制尚不十分明确的直肠和结肠慢性非特异性 炎性疾病。病因可能与遗传、免疫、环境等因素有关，目前临床上治疗轻-中 度 UC 诱导缓解的主要首选药物为氨基水杨酸制剂，由于柳氮磺胺吡啶较多的 副作用，美沙拉嗪制剂已逐渐成为首选药物。
但对于轻中度远段 UC 的治疗， 近年来局部治疗日益受到重视。我国 UC 的诊治共识意见提出远段 UC 由于病 变范围局部，可选择局部治疗（栓剂或灌肠剂）或全身治疗（口服制剂）的 5-ASA 制剂。
美沙拉嗪灌肠液是近年国外研制的局部溃疡性结肠炎治疗的 5-ASA 新型制剂，可以抑制炎性介质发生作用，降低炎症细胞的活化作用，使前列腺素 进行合成或者释放，减轻局部炎症。
市场及经济效益预测
根据 QYR（恒州博智）的统计及预测，2021年全球美沙拉嗪市场销售额达 到了1.6亿美元，预计2028年将达到2.3亿美元，年复合增长率（CAGR）为4.4% （ 2022-2028）。
    """
    
    try:
        print(f"\n测试文本: {test_text}")
        print("正在提取特征...")
        
        features = await g_feature_extractor.extract_all_features(test_text, "")
        
        print("✓ 特征提取成功")
        print(f"语义向量维度: {len(features['semantic_vector'])}")
        print(f"领域向量维度: {len(features['domain_vector'])}")
        print("\n提取的文本:")
        for dim, text in features['extracted_texts'].items():
            print(f"  {dim}: {text or '无'}")
        
        return True
    except Exception as e:
        print(f"✗ FeatureExtractor失败: {e}")
        return False

async def main():
    print("\nOne-API连接诊断工具\n")
    
    results = []
    
    results.append(("LLM连接", await test_llm_connection()))
    results.append(("Embedding连接", await test_embedding_connection()))
    results.append(("FeatureExtractor", await test_feature_extractor()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n所有测试通过！")
    else:
        print("\n部分测试失败，请检查配置。")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())
