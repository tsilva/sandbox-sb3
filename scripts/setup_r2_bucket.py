from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import boto3
from botocore.exceptions import ClientError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or smoke-test an S3/R2 checkpoint bucket")
    parser.add_argument("bucket", help="Bucket name, without s3://")
    parser.add_argument("--prefix", default="wandb/_smoke", help="Object prefix to smoke-test")
    parser.add_argument("--create", action="store_true", help="Create the bucket if it does not exist")
    parser.add_argument(
        "--keep-smoke-object",
        action="store_true",
        help="Leave the uploaded smoke-test object in the bucket",
    )
    return parser


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    args = build_parser().parse_args()
    endpoint_url = require_env("AWS_S3_ENDPOINT_URL")
    require_env("AWS_ACCESS_KEY_ID")
    require_env("AWS_SECRET_ACCESS_KEY")

    s3 = boto3.client("s3", endpoint_url=endpoint_url)

    try:
        s3.head_bucket(Bucket=args.bucket)
        print(f"bucket exists: s3://{args.bucket}")
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if not args.create or status not in {403, 404}:
            raise
        print(f"creating bucket: s3://{args.bucket}")
        s3.create_bucket(Bucket=args.bucket)

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    key = f"{args.prefix.strip('/')}/_smoke/{timestamp}.txt"
    body = f"r2 smoke test {timestamp}\n".encode()
    s3.put_object(Bucket=args.bucket, Key=key, Body=body, ContentType="text/plain")
    response = s3.get_object(Bucket=args.bucket, Key=key)
    downloaded = response["Body"].read()
    if downloaded != body:
        raise SystemExit("Smoke object round trip failed")

    print(f"smoke object round trip ok: s3://{args.bucket}/{key}")
    if args.keep_smoke_object:
        return
    try:
        s3.delete_object(Bucket=args.bucket, Key=key)
        print("smoke object deleted")
    except ClientError as exc:
        print(f"warning: could not delete smoke object: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
