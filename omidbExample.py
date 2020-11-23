import omidb


#db = omidb.DB('./omi-db', clients=['demd175588', 'demd175573'])
db = omidb.DB('/media/robert/ICEBERG_1/img/optimam/image_db', clients=['demd175588', 'demd175573'])
#db = omidb.DB('/media/robert/ICEBERG_1/img/optimam/image_db')
clients = [client for client in db]
[print(client.id) for client in clients]
