import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import requests
import json
from config.app_config import WEBUI_PORT, RESOURCE_TYPES

API_BASE_URL = "http://192.168.40.64:5310"

st.set_page_config(
    page_title="智能匹配引擎",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 智能匹配引擎")
st.sidebar.header("导航")

page = st.sidebar.radio(
    "功能菜单",
    ["匹配测试", "数据导入", "关于"],
    index=0
)

if page == "匹配测试":
    st.header("匹配测试")
    
    query = st.text_area(
        "输入查询文本",
        placeholder="例如：找上海做AI大模型的专家",
        height=100
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        resource_types = st.multiselect(
            "资源类型",
            options=RESOURCE_TYPES,
            default=["expert", "project"]
        )
    
    with col2:
        top_k = st.slider("返回数量", 1, 50, 10)
    
    with col3:
        use_hybrid = st.checkbox("启用混合检索", value=True)
    
    if st.button("开始匹配", type="primary"):
        if not query.strip():
            st.warning("请输入查询文本")
        else:
            with st.spinner("正在匹配..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/match",
                        json={
                            "query": query,
                            "resource_types": resource_types,
                            "top_k": top_k,
                            "use_hybrid": use_hybrid
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.subheader("解析结果")
                        parsed = result.get("query_parsed", {})
                        st.json(parsed)
                        
                        st.subheader("匹配结果")
                        results_by_type = result.get("results_by_type", {})
                        
                        for resource_type, items in results_by_type.items():
                            if items:
                                with st.expander(f"{resource_type} ({len(items)}条)"):
                                    for item in items:
                                        col_a, col_b = st.columns([1, 3])
                                        with col_a:
                                            st.metric("得分", f"{item.get('score', 0):.4f}")
                                        with col_b:
                                            st.write(f"**ID:** {item.get('id')}")
                                            st.write(f"**来源:** {item.get('source_id')}")
                                            if item.get('region'):
                                                st.write(f"**地域:** {item.get('region')}")
                                            if item.get('maturity'):
                                                st.write(f"**阶段:** {item.get('maturity')}")
                                            if item.get('metadata'):
                                                with st.expander("详细信息"):
                                                    st.json(item.get('metadata'))
                        
                        meta = result.get("meta", {})
                        st.info(f"共找到 {meta.get('total_candidates', 0)} 条结果，耗时 {meta.get('retrieval_time_ms', 0)}ms")
                    else:
                        st.error(f"请求失败: {response.text}")
                except Exception as e:
                    st.error(f"请求异常: {e}")

elif page == "数据导入":
    st.header("数据导入")
    
    resource_type = st.selectbox("选择资源类型", RESOURCE_TYPES)
    
    upload_type = st.radio("导入方式", ["手动输入", "JSON文件"])
    
    if upload_type == "手动输入":
        st.subheader("手动输入数据")
        
        item_id = st.text_input("数据ID")
        raw_text = st.text_area("向量化文本", height=150)
        
        with st.expander("扩展字段"):
            region = st.text_input("地域")
            maturity = st.selectbox("成熟度", ["", "研发", "小试", "中试", "量产"])
            custom_fields = st.text_area("其他字段(JSON格式)", "{}")
        
        if st.button("导入数据", type="primary"):
            if not item_id or not raw_text:
                st.warning("请填写ID和向量化文本")
            else:
                    try:
                        fields = json.loads(custom_fields) if custom_fields else {}
                        fields["region"] = region
                        fields["maturity"] = maturity
                        
                        data = {
                            "resource_type": resource_type,
                            "items": [{
                                "id": item_id,
                                "fields": fields,
                                "raw_text_for_embedding": raw_text
                            }]
                        }
                        
                        with st.spinner("正在导入数据..."):
                            response = requests.post(
                                f"{API_BASE_URL}/api/v1/data/insert",
                                json=data,
                                timeout=60
                            )
                        
                        st.subheader("导入结果")
                        if response.status_code == 200:
                            result = response.json()
                            ingested_count = result.get('ingested_count', 0)
                            failed_ids = result.get('failed_ids', [])
                            task_id = result.get('task_id')
                            
                            st.success(f"✅ 导入完成！成功: {ingested_count} 条，失败: {len(failed_ids)} 条")
                            
                            if task_id:
                                st.info(f"任务ID: {task_id}")
                            
                            if failed_ids:
                                st.warning(f"失败的ID: {', '.join(failed_ids)}")
                            
                            with st.expander("查看完整响应"):
                                st.json(result)
                        else:
                            st.error(f"❌ 导入失败 (状态码: {response.status_code})")
                            with st.expander("查看错误详情"):
                                st.json(response.json())
                    except json.JSONDecodeError:
                        st.error("❌ 其他字段JSON格式错误，请检查输入")
                    except requests.exceptions.Timeout:
                        st.error("❌ 请求超时，请检查API服务是否正常运行")
                    except requests.exceptions.ConnectionError:
                        st.error("❌ 无法连接到API服务，请检查服务是否启动")
                    except Exception as e:
                        st.error(f"❌ 导入异常: {str(e)}")
    
    else:
        st.subheader("JSON文件导入")
        uploaded_file = st.file_uploader("选择JSON文件", type=["json"])
        
        if uploaded_file:
            try:
                data = json.load(uploaded_file)
                st.json(data)
                
                if st.button("确认导入", type="primary"):
                    with st.spinner("正在导入..."):
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/data/insert",
                            json={"resource_type": resource_type, "items": data},
                            timeout=120
                        )
                    
                    st.subheader("导入结果")
                    if response.status_code == 200:
                        result = response.json()
                        ingested_count = result.get('ingested_count', 0)
                        failed_ids = result.get('failed_ids', [])
                        task_id = result.get('task_id')
                        
                        st.success(f"✅ 导入完成！成功: {ingested_count} 条，失败: {len(failed_ids)} 条")
                        
                        if task_id:
                            st.info(f"任务ID: {task_id}")
                        
                        if failed_ids:
                            st.warning(f"失败的ID: {', '.join(failed_ids)}")
                        
                        with st.expander("查看完整响应"):
                            st.json(result)
                    else:
                        st.error(f"❌ 导入失败 (状态码: {response.status_code})")
                        with st.expander("查看错误详情"):
                            st.json(response.json())
            except json.JSONDecodeError:
                st.error("❌ JSON文件格式错误，请检查文件内容")
            except requests.exceptions.Timeout:
                st.error("❌ 请求超时，请检查API服务是否正常运行")
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到API服务，请检查服务是否启动")
            except Exception as e:
                st.error(f"❌ 文件解析失败: {str(e)}")

elif page == "关于":
    st.header("关于")
    st.markdown("""
    **智能匹配引擎 (Intelligent Matching Engine)
    
    一个基于语义理解的智能匹配系统，支持：
    - 成果/专利/项目相似性检索
    - 需求与多类型资源匹配
    - 关键词增强搜索
    
    技术栈：
    - FastAPI + Streamlit
    - Milvus 向量数据库
    - BGE-M3 嵌入模型
    - Qwen3 大语言模型
    """)
    
    st.subheader("系统状态")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API服务正常运行")
        else:
            st.error("❌ API服务异常")
    except Exception as e:
        st.error(f"❌ 无法连接API服务")

st.sidebar.markdown("---")
st.sidebar.info(f"端口: {WEBUI_PORT}")
