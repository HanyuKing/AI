import json
import hashlib
import base64
import time
import uuid
import io
import yaml
import sqlparse
import qrcode
from datetime import datetime, timezone
from typing import Dict, Any, List

class LogicService:
    
    @staticmethod
    def format_json(content: str, indent: int = 4, sort_keys: bool = False) -> Dict[str, Any]:
        try:
            # Try parsing as JSON
            obj = json.loads(content)
            formatted = json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
            return {"result": formatted, "valid": True}
        except json.JSONDecodeError as e:
            return {"result": str(e), "valid": False}

    @staticmethod
    def json_to_yaml(content: str) -> Dict[str, Any]:
        try:
            obj = json.loads(content)
            yaml_str = yaml.dump(obj, allow_unicode=True, default_flow_style=False)
            return {"result": yaml_str, "valid": True}
        except Exception as e:
            return {"result": str(e), "valid": False}

    @staticmethod
    def format_sql(sql: str) -> str:
        return sqlparse.format(sql, reindent=True, keyword_case='upper')

    @staticmethod
    def calculate_hash(text: str, algorithm: str = "md5") -> str:
        data = text.encode('utf-8')
        if algorithm == "md5":
            return hashlib.md5(data).hexdigest()
        elif algorithm == "sha1":
            return hashlib.sha1(data).hexdigest()
        elif algorithm == "sha256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "sha512":
            return hashlib.sha512(data).hexdigest()
        else:
            raise ValueError("Unsupported algorithm")

    @staticmethod
    def base64_process(text: str, action: str = "encode") -> str:
        if action == "encode":
            return base64.b64encode(text.encode('utf-8')).decode('utf-8')
        else:
            try:
                return base64.b64decode(text).decode('utf-8')
            except Exception:
                return "Invalid Base64 String"

    @staticmethod
    def convert_timestamp(ts: float = None) -> Dict[str, Any]:
        if ts is None:
            ts = time.time()
        
        dt = datetime.fromtimestamp(ts)
        return {
            "timestamp": ts,
            "iso": dt.isoformat(),
            "local": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "utc": datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

    @staticmethod
    def convert_base(value: str, from_base: int, to_base: int) -> str:
        try:
            # Convert to decimal first
            decimal_value = int(value, from_base)
            
            if to_base == 10:
                return str(decimal_value)
            elif to_base == 2:
                return bin(decimal_value)[2:]
            elif to_base == 8:
                return oct(decimal_value)[2:]
            elif to_base == 16:
                return hex(decimal_value)[2:]
            else:
                # Simple implementation for other bases if needed, but usually 2,8,10,16 are enough
                return str(decimal_value)
        except ValueError:
            return "Invalid input for the given base"

    @staticmethod
    def generate_uuid(count: int = 1, uppercase: bool = False, hyphens: bool = True) -> List[str]:
        results = []
        for _ in range(count):
            u = str(uuid.uuid4())
            if not hyphens:
                u = u.replace("-", "")
            if uppercase:
                u = u.upper()
            results.append(u)
        return results

    @staticmethod
    def generate_qrcode(text: str, fill_color: str = "black", back_color: str = "white") -> bytes:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(text)
        qr.make(fit=True)

        img = qr.make_image(fill_color=fill_color, back_color=back_color)
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        return img_buffer.getvalue()

