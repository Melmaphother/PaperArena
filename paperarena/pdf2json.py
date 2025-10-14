# Copyright (c) Opendatalab. All rights reserved.
import copy
import json
import os
from pathlib import Path

from loguru import logger

from mineru.cli.common import (
    convert_pdf_bytes_to_bytes_by_pypdfium2,
    prepare_env,
    read_fn,
)
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
    union_make as pipeline_union_make,
)
from mineru.backend.pipeline.model_json_to_middle_json import (
    result_to_middle_json as pipeline_result_to_middle_json,
)
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.utils.models_download_utils import auto_download_and_get_model_root_path


def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="pipeline",  # The backend for parsing PDF, default is 'pipeline'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-sglang-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
            pipeline_doc_analyze(
                pdf_bytes_list,
                p_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        )

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(
                model_list,
                images_list,
                pdf_doc,
                image_writer,
                _lang,
                _ocr_enable,
                formula_enable,
            )

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            if f_draw_layout_bbox:
                draw_layout_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                )

            if f_draw_span_bbox:
                draw_span_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                )

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = pipeline_union_make(
                    pdf_info, f_make_md_mode, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = pipeline_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                md_writer.write_string(
                    f"{pdf_file_name}_model.json",
                    json.dumps(model_json, ensure_ascii=False, indent=4),
                )

            logger.info(f"local output dir is {local_md_dir}")
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        f_draw_span_bbox = False
        parse_method = "vlm"
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            pdf_file_name = pdf_file_names[idx]
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page_id, end_page_id
            )
            local_image_dir, local_md_dir = prepare_env(
                output_dir, pdf_file_name, parse_method
            )
            image_writer, md_writer = FileBasedDataWriter(
                local_image_dir
            ), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = vlm_doc_analyze(
                pdf_bytes,
                image_writer=image_writer,
                backend=backend,
                server_url=server_url,
            )

            pdf_info = middle_json["pdf_info"]

            if f_draw_layout_bbox:
                draw_layout_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf"
                )

            if f_draw_span_bbox:
                draw_span_bbox(
                    pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf"
                )

            if f_dump_orig_pdf:
                md_writer.write(
                    f"{pdf_file_name}_origin.pdf",
                    pdf_bytes,
                )

            if f_dump_md:
                image_dir = str(os.path.basename(local_image_dir))
                md_content_str = vlm_union_make(pdf_info, f_make_md_mode, image_dir)
                md_writer.write_string(
                    f"{pdf_file_name}.md",
                    md_content_str,
                )

            if f_dump_content_list:
                image_dir = str(os.path.basename(local_image_dir))
                content_list = vlm_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, image_dir
                )
                md_writer.write_string(
                    f"{pdf_file_name}_content_list.json",
                    json.dumps(content_list, ensure_ascii=False, indent=4),
                )

            if f_dump_middle_json:
                md_writer.write_string(
                    f"{pdf_file_name}_middle.json",
                    json.dumps(middle_json, ensure_ascii=False, indent=4),
                )

            if f_dump_model_output:
                model_output = ("\n" + "-" * 50 + "\n").join(infer_result)
                md_writer.write_string(
                    f"{pdf_file_name}_model_output.txt",
                    model_output,
                )

            logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
    path_list: list[Path],
    output_dir,
    lang="ch",
    backend="pipeline",
    method="auto",
    server_url=None,
    start_page_id=0,
    end_page_id=None,
):
    """
    Parameter description:
    path_list: List of document paths to be parsed, can be PDF or image files.
    output_dir: Output directory for storing parsing results.
    lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka']。
        Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
        Adapted only for the case where the backend is set to "pipeline"
    backend: the backend for parsing pdf:
        pipeline: More general.
        vlm-transformers: More general.
        vlm-sglang-engine: Faster(engine).
        vlm-sglang-client: Faster(client).
        without method specified, pipeline will be used by default.
    method: the method for parsing pdf:
        auto: Automatically determine the method based on the file type.
        txt: Use text extraction method.
        ocr: Use OCR method for image-based PDFs.
        Without method specified, 'auto' will be used by default.
        Adapted only for the case where the backend is set to "pipeline".
    server_url: When the backend is `sglang-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    start_page_id: Start page ID for parsing, default is 0
    end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
        )
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    import shutil
    import tempfile

    # args
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    pdf_files_dir = os.path.join(__dir__, "sampled_papers")
    output_dir = os.path.join(__dir__, "sampled_jsons")
    pdf_suffixes = [".pdf"]
    image_suffixes = [".png", ".jpeg", ".jpg"]

    # 收集所有PDF文件
    doc_path_list = []
    for doc_path in Path(pdf_files_dir).glob("*"):
        if doc_path.suffix in pdf_suffixes + image_suffixes:
            doc_path_list.append(doc_path)

    print(f"🚀 Found {len(doc_path_list)} PDF files to process")
    print(f"📂 Input directory: {pdf_files_dir}")
    print(f"📁 Output directory: {output_dir}")

    """如果您由于网络问题无法下载模型，可以设置环境变量MINERU_MODEL_SOURCE为modelscope使用免代理仓库下载模型"""
    # os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

    # 创建临时目录
    temp_dir = os.path.join(__dir__, "temp_pdf")
    os.makedirs(temp_dir, exist_ok=True)

    successful_count = 0
    failed_files = []

    print(f"📁 Created temporary directory: {temp_dir}")
    print(f"🔄 Processing files one by one to avoid multi-threading issues...")

    # 检查已处理的文件 (去掉.pdf后缀)
    processed_files = []
    if os.path.exists(output_dir):
        processed_files = os.listdir(output_dir)
    print(f"   🔄 Found {len(processed_files)} already processed files")

    try:
        # 逐个处理PDF文件
        for i, pdf_path in enumerate(doc_path_list, 1):
            try:
                # 获取不含后缀的文件名用于比较
                file_stem = pdf_path.stem  # 不含后缀的文件名

                # 跳过已处理的文件
                if file_stem in processed_files:
                    print(
                        f"   ⏭️ Skipping already processed file ({i}/{len(doc_path_list)}): {pdf_path.name}"
                    )
                    successful_count += 1  # 计入成功处理的数量
                    continue

                print(f"\n📄 Processing {i}/{len(doc_path_list)}: {pdf_path.name}")

                # 清空临时目录
                for temp_file in Path(temp_dir).glob("*"):
                    temp_file.unlink()

                # 复制当前PDF到临时目录
                temp_pdf_path = os.path.join(temp_dir, pdf_path.name)
                shutil.copy2(pdf_path, temp_pdf_path)
                print(f"   📋 Copied to temp directory: {temp_pdf_path}")

                # 处理临时目录中的单个PDF文件
                temp_doc_list = [Path(temp_pdf_path)]

                """Use pipeline mode with English language setting"""
                parse_doc(temp_doc_list, output_dir, lang="en", backend="pipeline")

                successful_count += 1
                print(f"   ✅ Successfully processed: {pdf_path.name}")

            except Exception as e:
                failed_files.append(str(pdf_path))
                print(f"   ❌ Failed to process {pdf_path.name}: {str(e)}")
                continue

    finally:
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🗑️ Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean up temporary directory: {e}")

    # 打印处理结果摘要
    print(f"\n" + "=" * 60)
    print(f"📊 PROCESSING SUMMARY")
    print(f"=" * 60)
    print(f"✅ Successfully processed: {successful_count}/{len(doc_path_list)} files")
    print(f"❌ Failed: {len(failed_files)} files")

    if failed_files:
        print(f"\n⚠️ Failed files:")
        for failed_file in failed_files:
            print(f"   - {Path(failed_file).name}")

        print(f"\n💡 Possible solutions for failed files:")
        print(f"   1. Try converting PDFs to a different format first")
        print(f"   2. Check if PDFs are password protected")
        print(f"   3. Try processing with different method (txt/ocr)")
        print(f"   4. Use a different backend (vlm-transformers)")

    print(f"\n🎉 Processing completed! Check output directory: {output_dir}")

    """To enable VLM mode, change the backend to 'vlm-xxx'"""
    # parse_doc(temp_doc_list, output_dir, lang="en", backend="vlm-transformers")  # more general.
    # parse_doc(temp_doc_list, output_dir, lang="en", backend="vlm-sglang-engine")  # faster(engine).
    # parse_doc(temp_doc_list, output_dir, lang="en", backend="vlm-sglang-client", server_url="http://127.0.0.1:30000")  # faster(client).
