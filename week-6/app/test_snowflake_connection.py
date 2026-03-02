# import snowflake.connector

# conn = snowflake.connector.connect(
#     account="SFEDU02-DCB73175",
#     user="GIRAFFE",
#     authenticator="externalbrowser",  # IMPORTANT
#     role="TRAINING_ROLE",
#     warehouse="WEATHER_TWIN_WH",
#     database="WEATHER_TWIN_DB",
#     schema="PUBLIC",
# )

# cur = conn.cursor()
# try:
#     cur.execute("""
#         SELECT
#             CURRENT_ACCOUNT(),
#             CURRENT_USER(),
#             CURRENT_ROLE(),
#             CURRENT_WAREHOUSE(),
#             CURRENT_DATABASE(),
#             CURRENT_SCHEMA()
#     """)
#     print(cur.fetchall())
# finally:
#     cur.close()
#     conn.close()

import snowflake.connector
import getpass

# Hard‑coded password (DON'T COMMIT THIS TO GIT)
BASE_PASSWORD = "Sn0wFl@ke@UMKC"  # your real password

totp = getpass.getpass("Enter your current MFA TOTP code: ")

conn = snowflake.connector.connect(
    account="SFEDU02-DCB73175",     # add region suffix if needed, e.g. ".us-east-1"
    user="GIRAFFE",
    password=BASE_PASSWORD,
    passcode=totp,                  # separate TOTP code
    authenticator="snowflake",
    role="TRAINING_ROLE",
    warehouse="WEATHER_TWIN_WH",
    database="WEATHER_TWIN_DB",
    schema="PUBLIC",
)

cur = conn.cursor()
cur.execute("SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_DATABASE(), CURRENT_SCHEMA()")
print(cur.fetchall())
cur.close()
conn.close()