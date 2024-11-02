# Haitian Creole Teaching Tools

---

## Running

### Using Python Interpreter

```shell
~ $ make run
```

### Using Docker

Development image:

```console
~ $ make build-dev
~ $ docker images --filter "label=name=blueprint"
REPOSITORY                                                             TAG                 IMAGE ID            CREATED             SIZE
docker.pkg.github.com/martinheinz/python-project-blueprint/blueprint   3492a40-dirty       acf8d09acce4        28 seconds ago      967MB
~ $ docker run acf8d09acce4
Hello World...
```

Production (Distroless) image:

```console
~ $ make build-prod VERSION=0.0.5
~ $ docker images --filter "label=version=0.0.5"
REPOSITORY                                                             TAG                 IMAGE ID            CREATED             SIZE
docker.pkg.github.com/martinheinz/python-project-blueprint/blueprint   0.0.5               65e6690d9edd        5 seconds ago       86.1MB
~ $ docker run 65e6690d9edd
Hello World...
```

## Testing

Test are ran every time you build _dev_ or _prod_ image. You can also run tests using:

```console
~ $ make test
```

## Cleaning

Clean _Pytest_ and coverage cache/files:

```console
~ $ make clean
```

Clean _Docker_ images:

```console
~ $ make docker-clean
```

## Creating Secret Tokens

Token is needed for example for _GitHub Package Registry_. To create one:

- Go to _Settings_ tab
- Click _Secrets_
- Click _Add a new secret_
  - _Name_: _name that will be accessible in GitHub Actions as `secrets.NAME`_
  - _Value_: _value_

### Resources

- [https://realpython.com/python-application-layouts/](https://realpython.com/python-application-layouts/)
- [https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6](https://dev.to/codemouse92/dead-simple-python-project-structure-and-imports-38c6)
- [https://github.com/navdeep-G/samplemod/blob/master/setup.py](https://github.com/navdeep-G/samplemod/blob/master/setup.py)
- [https://github.com/GoogleContainerTools/distroless/blob/master/examples/python3/Dockerfile](https://github.com/GoogleContainerTools/distroless/blob/master/examples/python3/Dockerfile)
