CREATE DATABASE IF NOT EXISTS fineract_tenants CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE DATABASE IF NOT EXISTS fineract_default CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- ensure root has full access from any host (docker network)
GRANT ALL PRIVILEGES ON fineract_tenants.* TO 'root'@'%' IDENTIFIED BY 'root';
GRANT ALL PRIVILEGES ON fineract_default.* TO 'root'@'%' IDENTIFIED BY 'root';
FLUSH PRIVILEGES;
