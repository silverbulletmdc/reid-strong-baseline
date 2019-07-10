import boto3
import os
import math
import argparse
import functools
import concurrent.futures

DEFAULT_SITE = 'hh-b'


def get_oss_server_addr(site):
    return 'http://oss.{}.brainpp.cn'.format(site)


def get_oss_client(site=DEFAULT_SITE):
    return boto3.client('s3', endpoint_url=get_oss_server_addr(site))


def load_from_oss(oss_client, bucket, path):
    resp = oss_client.get_object(Bucket=bucket, Key=path)
    return resp['Body'].read()


def save_to_oss(bucket, path):
    client = get_oss_client()
    object_name = os.path.basename(path)
    if os.path.isfile(path):
        client.upload_file(path, bucket, object_name)
    elif os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                key = os.path.join(object_name, f)
                file_path = os.path.join(path, f)
                client.upload_file(file_path, bucket, key)


class S3MultipartUpload(object):
    def __init__(self,
                 url,
                 bucket,
                 key,
                 local_path,
                 part_size=64 * 1024 * 1024,
                 max_workers=10):
        self.bucket = bucket
        self.key = key
        self.path = local_path
        self.total_bytes = os.stat(local_path).st_size
        self.part_bytes = part_size
        self.num_parts = math.ceil(self.total_bytes / self.part_bytes)
        self.s3_client = boto3.client('s3', endpoint_url=url)
        self.executor_cls = concurrent.futures.ThreadPoolExecutor
        self.max_workers = max_workers

    def list_parts(self, upload_id):
        parts = []
        mpus = self.s3_client.list_parts(Bucket=self.bucket, Key=self.key, UploadId=upload_id)
        if "Parts" in mpus:
            for u in mpus["Parts"]:
                parts.append({"PartNumber": u["PartNumber"], "ETag": u["ETag"]})
        return parts

    def list_multipart_uploads(self):
        mpus = self.s3_client.list_multipart_uploads(Bucket=self.bucket, Prefix=self.key)
        return mpus

    def abort_all(self):
        mpus = self.s3_client.list_multipart_uploads(Bucket=self.bucket)
        aborted = []
        if "Uploads" in mpus:
            for u in mpus["Uploads"]:
                upload_id = u["UploadId"]
                aborted.append(
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket, Key=self.key, UploadId=upload_id))
        return aborted

    def create(self):
        mpu = self.s3_client.create_multipart_upload(Bucket=self.bucket, Key=self.key)
        mpu_id = mpu["UploadId"]
        return mpu_id

    def continue_upload(self, mpu_id):
        parts = []
        uploaded_parts_num = []
        needed_upload_parts_num = []
        uploaded_parts = self.list_parts(mpu_id)
        for item in uploaded_parts:
            uploaded_parts_num.append(item["PartNumber"])
        for num in range(1, self.num_parts + 1):
            if num not in uploaded_parts_num:
                needed_upload_parts_num.append(num)
        with self.executor_cls(max_workers=self.max_workers) as executor:
            upload_partial = functools.partial(self._upload_one_part, mpu_id)
            for part in executor.map(upload_partial, needed_upload_parts_num):
                parts.append(part)
        parts += uploaded_parts
        return parts

    def upload(self, mpu_id):
        parts = []
        with self.executor_cls(max_workers=self.max_workers) as executor:
            upload_partial = functools.partial(self._upload_one_part, mpu_id)
            for part in executor.map(upload_partial, range(1, self.num_parts + 1)):
                parts.append(part)
        return parts

    def _upload_one_part(self, mpu_id, part_number):
        with open(self.path, "rb") as f:
            offset = self.part_bytes * (part_number - 1)
            max_chunk_size = self.total_bytes - offset
            chunk_size = min(max_chunk_size, self.part_bytes)
            f.seek(offset)
            data = f.read(chunk_size)
            part = self.s3_client.upload_part(
                Body=data, Bucket=self.bucket, Key=self.key,
                UploadId=mpu_id, PartNumber=part_number)
            return {"PartNumber": part_number, "ETag": part["ETag"]}

    def complete(self, mpu_id, parts):
        result = self.s3_client.complete_multipart_upload(
            Bucket=self.bucket,
            Key=self.key,
            UploadId=mpu_id,
            MultipartUpload={"Parts": parts})
        return result


def parse_args():
    parser = argparse.ArgumentParser(description='Multipart upload')
    parser.add_argument('--url', required=True)
    parser.add_argument('--bucket', required=True)
    parser.add_argument('--key', required=True)
    parser.add_argument('--path', required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mpu = S3MultipartUpload(
        args.url,
        args.bucket,
        args.key,
        args.path)
    mpu.abort_all()
    mpu_id = mpu.create()
    parts = mpu.upload(mpu_id)
    print(mpu.complete(mpu_id, parts))

    # continue upload
    # parts = mpu.continue_upload(mpu_id)
    # print(mpu.complete(mpu_id, parts))
