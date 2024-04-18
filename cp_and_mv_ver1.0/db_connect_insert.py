
import pymysql

# MySQL 서버 연결 정보
db_config = {
    'host': '10.10.52.141',
    'port':3306,
    'user': 'jsum',
    'password': '11110000',
    'database': 'dance_record'
}
try:
    conn = pymysql.connect(**db_config)
    # MySQL 서버에 연결
    

    if conn.open:
        print('MySQL 서버에 연결되었습니다.')
    else:
        print('MySQL 서버에 연결할 수 없습니다.')
        
        
        ###########  기록 삽입
        # 데이터 삽입을 위한 SQL 쿼리
    insert_query = "INSERT INTO dance_score (id, score) VALUES (%s, %s)"
    update_query = "UPDATE dance_score SET score = %s WHERE id = %s"

    # 데이터 삽입할 값
    data_to_insert = ('test1', '101')

    # 커서 생성
    cursor = conn.cursor()

    # 쿼리 실행 (인서트)
    # cursor.execute(insert_query, data_to_insert)
    new_score = 95  # 새로운 점수
    username = 'test1'  # 업데이트할 사용자 이름

    # 쿼리 실행
    cursor.execute(update_query, (new_score, username))

    # 변경사항 커밋
    conn.commit()

    print(f'{cursor.rowcount}개의 레코드가 삽입되었습니다.')

# 커서 및 연결 닫기
    cursor.close()
    conn.close()
except pymysql.Error as e:
    print(f'MySQL 에러 발생: {e}')

# # 데이터 업데이트를 위한 SQL 쿼리
# update_query = "UPDATE your_table_name SET column1 = %s WHERE column2 = %s"

# # 데이터 업데이트할 값
# new_value = 'new_value'
# condition_value = 'condition_value'

# # 커서 생성
# cursor = conn.cursor()

# # 쿼리 실행
# cursor.execute(update_query, (new_value, condition_value))

# # 변경사항 커밋
# conn.commit()

# print(f'{cursor.rowcount}개의 레코드가 업데이트되었습니다.')

# # 커서 및 연결 닫기
# cursor.close()
# conn.close()
# import mysql.connector

# def create_connection():
#     # MySQL 서버 연결 정보
#     db_config = {
#         'host': 'your_host',
#         'user': 'your_username',
#         'password': 'your_password',
#         'database': 'your_database_name'
#     }
    
#     # MySQL 서버에 연결
#     conn = mysql.connector.connect(**db_config)
    
#     if conn.is_connected():
#         print('MySQL 서버에 연결되었습니다.')
#     else:
#         print('MySQL 서버에 연결할 수 없습니다.')
    
#     return conn

# def register_user(username, password):
#     conn = create_connection()
#     cursor = conn.cursor()
    
#     # 사용자 등록을 위한 SQL 쿼리
#     insert_query = "INSERT INTO members (username, password) VALUES (%s, %s)"
    
#     try:
#         # 사용자 등록
#         cursor.execute(insert_query, (username, password))
#         conn.commit()
#         print(f'{username} 님, 회원 가입이 완료되었습니다.')
#     except mysql.connector.Error as err:
#         print(f"회원 가입 중 오류 발생: {err}")
    
#     cursor.close()
#     conn.close()

# def login_user(username, password):
#     conn = create_connection()
#     cursor = conn.cursor()
    
#     # 사용자 조회를 위한 SQL 쿼리
#     select_query = "SELECT id FROM members WHERE username = %s AND password = %s"
    
#     try:
#         # 사용자 조회
#         cursor.execute(select_query, (username, password))
#         member = cursor.fetchone()
        
#         if member:
#             print(f'{username} 님, 로그인 성공!')
#         else:
#             print('잘못된 사용자 이름 또는 비밀번호입니다.')
#     except mysql.connector.Error as err:
#         print(f"로그인 중 오류 발생: {err}")
    
#     cursor.close()
#     conn.close()

# # 회원 가입 및 로그인 테스트
# if __name__ == "__main__":
#     # 새로운 사용자 등록
#     register_user('alice', 'password123')
    
#     # 로그인 시도
#     login_user('alice', 'password123')
