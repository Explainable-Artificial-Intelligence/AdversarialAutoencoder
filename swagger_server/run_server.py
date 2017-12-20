import connexion


if __name__ == '__main__':

    app = connexion.App(__name__)
    app.add_api('swagger/swagger.yaml')
    application = app.app

    # run our server
    app.run(port=8080)