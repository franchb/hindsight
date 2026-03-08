#!/bin/bash
# init-vchord.sh — Initializes VectorChord extensions and tokenizer for Hindsight.
#
# Runs as a docker-entrypoint-initdb.d script (executes once on first DB init).
# Handles both single-instance (default POSTGRES_DB) and multi-instance setups
# via EXTRA_DATABASES env var.
#
# EXTRA_DATABASES format: dbname:user:password,dbname2:user2:password2,...

set -euo pipefail

install_extensions() {
    local db="$1"
    echo "Installing VectorChord extensions in database '$db'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
        CREATE EXTENSION IF NOT EXISTS vchord CASCADE;
        CREATE EXTENSION IF NOT EXISTS pg_tokenizer CASCADE;
        CREATE EXTENSION IF NOT EXISTS vchord_bm25 CASCADE;
EOSQL
    echo "Creating llmlingua2 tokenizer in database '$db'..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
        SELECT create_tokenizer('llmlingua2', \$\$ model = "llmlingua2" \$\$);
EOSQL
    echo "Extensions and tokenizer ready in '$db'."
}

# 1. Install extensions in the default POSTGRES_DB
install_extensions "$POSTGRES_DB"

# 2. Process EXTRA_DATABASES if set
if [ -n "${EXTRA_DATABASES:-}" ]; then
    IFS=',' read -ra DB_ENTRIES <<< "$EXTRA_DATABASES"
    for entry in "${DB_ENTRIES[@]}"; do
        IFS=':' read -r db_name db_user db_pass <<< "$entry"

        if [ -z "$db_name" ] || [ -z "$db_user" ] || [ -z "$db_pass" ]; then
            echo "ERROR: Invalid EXTRA_DATABASES entry: '$entry' (expected dbname:user:password)"
            exit 1
        fi

        echo "Creating user '$db_user' and database '$db_name'..."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
            DO \$\$
            BEGIN
                IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${db_user}') THEN
                    CREATE ROLE ${db_user} WITH LOGIN PASSWORD '${db_pass}';
                END IF;
            END
            \$\$;
            CREATE DATABASE ${db_name} OWNER ${db_user};
EOSQL

        install_extensions "$db_name"

        echo "Granting privileges to '$db_user' on '$db_name'..."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db_name" <<-EOSQL
            GRANT ALL PRIVILEGES ON DATABASE ${db_name} TO ${db_user};
            GRANT ALL ON SCHEMA public TO ${db_user};
            GRANT USAGE ON SCHEMA bm25_catalog TO ${db_user};
            GRANT SELECT ON ALL TABLES IN SCHEMA bm25_catalog TO ${db_user};
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO ${db_user};
            ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO ${db_user};
EOSQL
        echo "Database '$db_name' ready for user '$db_user'."
    done
fi

echo "VectorChord initialization complete."
