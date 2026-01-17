from ui.CouponAnalysisGUI import CouponAnalysisGUI
from service.DataProcessor import DataProcessor

# =========================================================
#EXECUÇÃO PRINCIPAL
# =========================================================
if __name__ == "__main__":
    #processa dados
    processor = DataProcessor('in-vehicle-coupon-recommendation.csv')
    processor.process_data()

    #inicia a interface
    app = CouponAnalysisGUI(processor)
    app.run()