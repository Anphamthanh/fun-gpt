from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from langserve import add_routes
import logging
import json

from app.entities.apis import TickerReqDto
from app.utils.analyze_ticker import generate_financial_report
from app.utils.misc import NumpyEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FinGPT")

app = FastAPI()


@app.post("/v1/reports/ticker")
async def gen_ticker_report(req: TickerReqDto):
    logger.info(f'Received request for sympol: {req.symbol}')
    try:
        report = generate_financial_report(
            req.symbol,
            req.sec_api_key,
            req.llm_api_key,
        )
        report_json = json.dumps(report, cls=NumpyEncoder)
        return JSONResponse(content=json.loads(report_json))
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Edit this to add the chain you want to add
# add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
