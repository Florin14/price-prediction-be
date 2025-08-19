from project_helpers import init_logger

try:
    import logging
    from fastapi.middleware.cors import CORSMiddleware
    from extensions import run_api, api, init_db, DBSessionMiddleware, SqlBaseModel
    from modules import (
        userRouter,
        listingRouter,
        locationRouter,
        priceHistoryRouter,
        predictionRouter,
        clientRouter,
        authRouter,
    )
    from project_helpers.schemas import ErrorSchema

except Exception as e:
    logging.error(str(e))
    exit(1)


def startup():
    init_logger()
    init_db()


#     # init_scheduler(engine, starter_callback=auto_jobs_starter)
#
#
# def shutdown():
#     pass
# scheduler.shutdown()

if __name__ == "__main__":
    try:
        api.add_event_handler("startup", startup)
        api.add_event_handler("shutdown", startup)
        api.add_middleware(DBSessionMiddleware)
        api.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "https://price-prediction-fe.onrender.com"
            ],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        run_api(
            host="localhost",
            port=8002,
            routers=[
                userRouter,
                listingRouter,
                locationRouter,
                priceHistoryRouter,
                predictionRouter,
                clientRouter,
                authRouter,
            ],
            responses={
                500: {"model": ErrorSchema},
                401: {"model": ErrorSchema},
                422: {"model": ErrorSchema},
                404: {"model": ErrorSchema},
            },
        )
    except Exception as e:
        logging.error(str(e))
        exit(1)
