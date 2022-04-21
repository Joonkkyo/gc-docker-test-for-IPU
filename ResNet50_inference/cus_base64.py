from typing import Any
import base64


from pydantic import PydanticTypeError, BaseModel

class Base64Error(PydanticTypeError):
    msg_template = 'value is not valid base64'

class Base64Bytes(bytes):
    @classmethod
    def encode(cls, data: bytes) -> 'Base64Bytes':
        return Base64Bytes(base64.b64encode(data))

    def decode(self) -> bytes:
        return self._decoded_bytes

    def decode_str(self) -> str:
        return self._decoded_bytes.decode()


    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: Any) -> 'Base64Bytes':
        if isinstance(value, int):
            raise Base64Error

        if isinstance(value, (bytes, str, bytearray, memoryview)):
            try:
                base64.b64decode(value, validate=True)
            except ValueError as e:
                raise Base64Error from e
            
            return Base64Bytes(bytes(value,encoding='utf8'))

        try:
            encoded = base64.b64encode(bytes(value,encoding='utf8'))
            return Base64Bytes(encoded)
        except ValueError as e:
            raise Base64Error from e

# ##### Basic tests #####

class B64Model(BaseModel):
    encoded: Base64Bytes