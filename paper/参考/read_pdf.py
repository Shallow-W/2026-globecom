import sys
import os

# Try different PDF libraries
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        print("请先安装: pip install PyPDF2 或 pip install pypdf")
        sys.exit(1)


def read_pdf(pdf_path, password=None):
    """读取PDF文件内容"""
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return

    try:
        reader = PdfReader(pdf_path)

        # 如果PDF有密码保护
        if reader.is_encrypted:
            if password:
                reader.decrypt(password)
            else:
                # 尝试空密码
                reader.decrypt("")
                if reader.is_encrypted:
                    print(f"PDF文件已加密，请提供密码")
                    print(f"用法: python read_pdf.py <pdf_path> [password]")
                    return

        # 读取所有页面
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                # 尝试修复编码问题
                try:
                    text = text.encode('gbk', errors='ignore').decode('gbk', errors='ignore')
                except:
                    pass
            print(f"\n{'='*60}")
            print(f"第 {i+1} 页")
            print(f"{'='*60}")
            print(text if text else "[此页无文本内容]")

    except Exception as e:
        print(f"读取错误: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python read_pdf.py <pdf_path> [password]")
        print("\n示例:")
        print("  python read_pdf.py GSTC.pdf")
        print("  python read_pdf.py 'GSTC(MMC)_1.23.pdf' 123456")
    else:
        pdf_path = sys.argv[1]
        password = sys.argv[2] if len(sys.argv) > 2 else None
        read_pdf(pdf_path, password)
