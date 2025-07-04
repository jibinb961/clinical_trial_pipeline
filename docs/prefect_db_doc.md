Prefect server overview
Self-host your own Prefect server instance.
After installing Prefect, you have a Python SDK client that can communicate with either Prefect Cloud or a self-hosted Prefect server, backed by a database and a UI.

Prefect Cloud and self-hosted Prefect server share a common set of capabilities. Prefect Cloud provides the additional features required by organizations such as RBAC, Audit logs, and SSO. See the Prefect Cloud overview for more information.

We recommend using the same version of prefect for the client and server. Old clients are compatible with new servers, but new clients can be incompatible with old servers as new fields are added to the REST API. The server will typically return a 422 status code if any issues occur.
​
The Prefect database

The Prefect database persists data to track the state of your flow runs and related Prefect concepts, including:

Flow run and task run state
Run history
Logs
Deployments
Flow and task run concurrency limits
Storage blocks for flow and task results
Variables
Artifacts
Work pool status
Prefect supports the following databases:

SQLite (default in Prefect): Recommended for lightweight, single-server deployments. SQLite requires essentially no setup.
PostgreSQL: Best for connecting to external databases, but requires additional setup (such as Docker). Prefect uses the pg_trgm extension, so it must be installed and enabled.
​
Using the database

A local SQLite database is the default database and is configured upon Prefect installation. The database is located at ~/.prefect/prefect.db by default.

To reset your database, run the CLI command:


Copy
prefect server database reset -y
This command clears all data and reapplies the schema.

​
Database settings

Prefect provides several settings for configuring the database. The default settings are:


Copy
PREFECT_API_DATABASE_CONNECTION_URL='sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db'
PREFECT_API_DATABASE_ECHO='False'
PREFECT_API_DATABASE_MIGRATE_ON_START='True'
PREFECT_API_DATABASE_PASSWORD='None'
Save a setting to your active Prefect profile with prefect config set.

​
Configure a PostgreSQL database

Connect Prefect to a PostgreSQL database by setting the following environment variable:


Copy
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"
The above environment variable assumes:

You have a username called postgres
Your password is set to yourTopSecretPassword
Your database runs on the same host as the Prefect server instance, localhost
You use the default PostgreSQL port 5432
Your PostgreSQL instance has a database called prefect
​
Quickstart: configure a PostgreSQL database with Docker

Start a PostgreSQL instance to use as your Prefect database with the following command (which starts a Docker container running PostgreSQL):


Copy
docker run -d --name prefect-postgres -v prefectdb:/var/lib/postgresql/data -p 5432:5432 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=yourTopSecretPassword -e POSTGRES_DB=prefect postgres:latest
The above command:

Pulls the latest version of the official postgres Docker image, which is compatible with Prefect.
Starts a container with the name prefect-postgres.
Creates a database prefect with a user postgres and yourTopSecretPassword password.
Mounts the PostgreSQL data to a Docker volume called prefectdb to provide persistence if you ever have to restart or rebuild that container.
Run the command below to set your current Prefect Profile to the PostgreSQL database instance running in your Docker container.


Copy
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"
​
Confirm your PostgreSQL database configuration

Inspect your Prefect profile to confirm that the environment variable has been properly set:


Copy
prefect config view --show-sources

Copy
You should see output similar to the following:

PREFECT_PROFILE='my_profile'
PREFECT_API_DATABASE_CONNECTION_URL='********' (from profile)
PREFECT_API_URL='http://127.0.0.1:4200/api' (from profile)
Start the Prefect server to use your PostgreSQL database instance:


Copy
prefect server start
​
In-memory database

To use an in-memory SQLite database, set the following environment variable:


Copy
prefect config set PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///file::memory:?cache=shared&uri=true&check_same_thread=false"
Use SQLite database for testing only

SQLite does not support multiprocessing. For high orchestration volume, use PostgreSQL.
​
Migrations

Prefect uses Alembic to manage database migrations. Alembic is a database migration tool to use with the SQLAlchemy Database Toolkit for Python. Alembic provides a framework for generating and applying schema changes to a database.

Apply migrations to your database with the following commands:

To upgrade:


Copy
prefect server database upgrade -y
To downgrade:


Copy
prefect server database downgrade -y
Use the -r flag to specify a specific migration version to upgrade or downgrade to. For example, to downgrade to the previous migration version, run:


Copy
prefect server database downgrade -y -r -1
or to downgrade to a specific revision:


Copy
prefect server database downgrade -y -r d20618ce678e
To downgrade all migrations, use the base revision.

See the contributing docs to learn how to create a database migration.

​
Prefect server installation notes

Your self-hosted server must meet the following requirements and configuration settings.

​
SQLite

SQLite is not packaged with the Prefect installation. But most systems already have SQLite installed, and it is typically bundled with Python.

If you self-host a Prefect server instance with a SQLite database, certain Linux versions of SQLite can be problematic. Compatible versions include Ubuntu 22.04 LTS and Ubuntu 20.04 LTS.

To confirm SQLite is installed, run:


Copy
sqlite3 --version
​
Use a self-signed SSL certificate

When using a self-signed SSL certificate, you need to configure your environment to trust the certificate. Add the certificate to your system bundle and point your tools to use that bundle by configuring the SSL_CERT_FILE environment variable.

If the certificate is not part of your system bundle, set the PREFECT_API_TLS_INSECURE_SKIP_VERIFY to True to disable certificate verification altogether.

Disabling certificate validation is insecure and only suggested as an option for testing.
​
Reverse proxy

Here is an example of basic Nginx configuration if you want to host a Prefect server behind a reverse proxy:


Copy
server {
    listen 80;
    listen [::]:80;
    server_name prefect.example.com; # the domain name you set up to access your Prefect server from outside
    location / {
        return 301 https://$host$request_uri; # HTTP requests are forwarded to HTTPS
    }
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name prefect.example.com; # the domain name you set up to access your Prefect server from outside

    ssl_certificate /path/to/ssl/certificate.pem;
    ssl_certificate_key /path/to/ssl/certificate_key.pem;

    location /api { # API routes
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;

        # for websocket communication with prefect client
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # for basic authentication
        proxy_set_header Authorization $http_authorization;
        proxy_pass_header Authorization;

        proxy_pass  http://127.0.0.1:4200; # port is 4200 or value of Prefect server API port setting
    }

    location / {
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_pass http://127.0.0.1:4200;  # port is 4200 or value of Prefect server API port setting
    }
}
​
Proxies

Prefect supports communicating with proxies through environment variables. Whether you use a Prefect Cloud account or self-host a Prefect server instance, set HTTPS_PROXY and SSL_CERT_FILE in your environment. Then the underlying network libraries will route Prefect’s requests appropriately.

Alternatively, the Prefect library connects to the API through any proxies you have listed in the HTTP_PROXY or ALL_PROXY environment variables. You may also use the NO_PROXY environment variable to specify which hosts should not pass through the proxy.

For more information about these environment variables, see the cURL documentation.

​
UI

When self-hosting the UI behind a proxy or reverse proxy, there are a few settings you may need to keep in mind:

PREFECT_UI_API_URL: the connection URL for communication from the UI to the API
PREFECT_UI_SERVE_BASE: the base URL path to serve the UI from
PREFECT_UI_URL: this is a convenience setting for clients that print or log UI URLs (for example, the Prefect CLI)