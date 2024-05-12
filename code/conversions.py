#Things that we need to approx, maybe even falsely
idiots = {
          "MSFT":["AccountsPayableAndAccruedLiabilitiesCurrent"]
          }

#Different names
measure_conversion = {"Assets":["EquityAndLiabilities", "LiabilitiesAndStockholdersEquity"],
                    "AssetsCurrent":["CurrentAssets"],
                    "LiabilitiesCurrent":["CurrentLiabilities"],
                    "AssetsNoncurrent":["NoncurrentAssets"],
                    "LiabilitiesNoncurrent":["NoncurrentLiabilities"],
                    "AccountsPayableCurrent": ["AccountsPayableTradeCurrent"],
                    "EntityCommonStockSharesOutstanding": ["CommonStockSharesOutstanding","NumberOfSharesOutstanding"],
                    "StockholdersEquity":["Equity","StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
                    "DerivativeLiabilities":["FairValueLiabilitiesMeasuredOnRecurringBasisDerivativeFinancialInstrumentsLiabilities"],
                    "DerivativeAssets":["FairValueAssetsMeasuredOnRecurringBasisDerivativeFinancialInstrumentsAssets"],
                    "DepreciationAndAmortization": ["DepreciationDepletionAndAmortization"],
                    "ShortTermBorrowings":["DebtCurrent"],
                    "CostofRevenue":["CostOfGoodsAndServicesSold"],
                    "CostOfGoodsAndServicesSold": ["CostOfRevenue","CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
                    "CostOfGoodsSold": ["CostOfGoodsSoldExcludingDepreciationDepletionAndAmortization"],
                    "Amortization": ["FiniteLivedIntangibleAssetsAmortizationExpenseNextTwelveMonths","AmortizationOfIntangibleAssets"],
                    "IncomeTaxesPaidNet":["IncomeTaxesPaid"],
                    "NetIncomeLoss": ["ProfitLoss"],
                    "OperatingExpenses": ["OperatingCostsAndExpenses"],
                    "PaymentsToAcquirePropertyPlantAndEquipment": ["PurchasesOfPropertyPlantAndEquipment", "PaymentsForPropertyPlantAndEquipment"],
                    "NetCashProvidedByUsedInOperatingActivities" : ["CashAndCashEquivalentsFromOperatingActivities", "OperatingActivitiesNetCashInflowsOutflows","CashFlowsFromOperatingActivities"],
                    "InterestIncome" : ["InvestmentIncomeInterest"],
                    "LongTermDebtCurrent" : ["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"],
                    "CapitalExpenditure" : ["PaymentsToAcquireProductiveAssets","PaymentsForCapitalImprovements"],
                    "InventoryNet" : ["InventoryFinishedGoods"]
}
approximate_measure_conversion = {
                    "Liabilities":["LiabilitiesFairValueDisclosure"],
                    "AssetsCurrent":["CurrentAssetsOtherThanAssetsOrDisposalGroupsClassifiedAsHeldForSaleOrAsHeldForDistributionToOwners",],
                    "LiabilitiesCurrent":["CurrentLiabilitiesOtherThanLiabilitiesIncludedInDisposalGroupsClassifiedAsHeldForSale"],
                    "EntityCommonStockSharesOutstanding": ["WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding"],
                    "PrepaidExpenseAndOtherAssetsCurrent":["OtherAssetsCurrent"],
                    "CapitalLeaseObligationsNoncurrent": ["OperatingLeaseLiabilityNoncurrent"],
                    "CapitalLeaseObligationsCurrent": ["LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths"],
                    "IntangibleAssetsNetExcludingGoodwill":["FiniteLivedIntangibleAssetsNet"],
                    "AccountsReceivableNet": ["AccountsReceivableNetCurrent"],
                    "Revenues": ["SalesRevenueNet","RevenueFromContractWithCustomerExcludingAssessedTax"],
                    "CostOfGoodsSold":["CostOfGoodsAndServicesSold", "CostOfRevenue"],
                    "CostofRevenue":["CostOfGoodsSold"],
                    "CostOfGoodsAndServicesSold": ["CostOfGoodsSold"],
                    "AccountsPayableAndAccruedLiabilities": ["AccountsPayable"],
                    "DepreciationAndAmortization":["Depreciation"],
                    "ShortTermBorrowings":["LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths","LongTermDebtCurrent"],
                    "AccountsReceivableNet" :["AccountsReceivableGrossCurrent"],
                    "PaymentsToAcquireProductiveAssets": ["PaymentsToAcquirePropertyPlantAndEquipment"],
                    "LongTermDebt" : ["LongTermDebtAndCapitalLeaseObligations"],
                    "AccountsPayableCurrent": ["AccountsPayable"],
                    "InventoryNet": ["InventoryGross"],
                    "NetCashProvidedByUsedInOperatingActivities":["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"]
}
additive_conversion = {"Assets":["AssetsCurrent", "AssetsNoncurrent"],
                    "Liabilities":["LiabilitiesCurrent", "LiabilitiesNoncurrent"],
                    "AccountsPayableAndAccruedLiabilitiesCurrent":["AccountsPayableCurrent", "AccruedLiabilitiesCurrent"],
                    "CashCashEquivalentsAndShortTermInvestments": ["CashAndCashEquivalentsAtCarryingValue","ShortTermInvestments"],
                    "AccountsReceivableNet": ["AccountsReceivableNetCurrent", "AccountsReceivableNetNoncurrent"],
                    "CostsAndExpenses":["OperatingExpenses", "CostOfGoodsAndServicesSold"],
                    "GrossProfit":["OperatingIncomeLoss","OperatingExpenses"],
                    "AccountsPayableAndAccruedLiabilities": ["AccruedLiabilities", "AccountsPayable"],
                    "LongTermDebt": ["LongTermDebtCurrent","LongTermDebtNoncurrent"],
                    "DepreciationAndAmortization": ["Depreciation", "AmortizationOfIntangibleAssets"],
                    "OperatingIncomeLoss":["NetIncomeLoss","IncomeTaxesPaidNet", "DepreciationAndAmortization","InterestExpense"],
                    "CapitalExpenditure":["PaymentsToAcquireEquipmentOnLease","PaymentsToAcquireOilAndGasPropertyAndEquipment","PaymentsToAcquireOtherProductiveAssets","PaymentsToAcquireProductiveAssets"],
                    "Revenues": ["GrossProfit", "CostOfGoodsAndServicesSold"],
                    "InterestExpense": ["InterestPaidNet", "InterestIncome"],
                    "NetCashProvidedByUsedInOperatingActivities": ["NetCashProvidedByUsedInOperatingActivitiesContinuingOperations","NetCashProvidedByUsedInDiscontinuedOperations"],
}       
approximate_additive_conversion = {
                    "LiabilitesCurrent":(["ShortTermBorrowings","AccountsPayableAndAccruedLiabilitiesCurrent","TaxesPayableCurrent", "DividendsPayableCurrent","OtherLiabilitiesCurrent"], 4),
                    "AssetsCurrent":(["CashCashEquivalentsAndShortTermInvestments","AccountsReceivableNetCurrent", "CapitalLeaseObligationsCurrent","InventoryNet","PrepaidExpenseAndOtherAssetsCurrent"], 5),
                    "LiabilitesNoncurrent":(["LongTermDebtNoncurrent","CapitalLeaseObligationsNoncurrent","DeferredTaxLiabilities","PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent","CapitalLeaseObligationsNoncurrent","DeferredRevenue","OtherLiabilitiesNoncurrent"], 5),
                    "AssetsNoncurrent": (["PropertyPlantAndEquipmentNet","IntangibleAssetsNetExcludingGoodwill","AccountsReceivableNetNoncurrent","OtherAssetsNoncurrent"],4), #FINISH WITH GPT-4
                    "OtherThanInventoryCurrent": (["CashAndCashEquivalentsAtCarryingValue","AccountsReceivableNetCurrent", "PrepaidExpenseAndOtherAssetsCurrent","DeferredTaxAssetsNetCurrent", "AvailableForSaleSecuritiesCurrent","AvailableForSaleSecuritiesDebtSecuritiesCurrent", "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedCurrentAssets", "TradingSecuritiesCurrent","DepositsAssetsCurrent","AssetsHeldForSaleCurrent"],7),
                    "PrepaidExpenseAndOtherAssetsCurrent": (["PrepaidExpenseCurrent","OtherPrepaidExpenseCurrent", "OtherAssetsCurrent"],3),
                    "ShortTermBorrowings": (["LongTermDebtCurrent", "CommercialPaper", "LineOfCreditFacilityAmountOutstanding", "FinanceLeaseLiabilityPaymentsRemainderOfFiscalYear", "FinanceLeaseLiabilityPaymentsDueNextTwelveMonths"],3),
                    "LongTermDebt": (["LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive","LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFive", "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearFour","LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree","LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo","LongTermDebtMaturitiesRepaymentsOfPrincipalInNextTwelveMonths"], 6),
                    "AccountsPayableAndAccruedLiabilitiesCurrent": (["AccountsPayableCurrent", "AccruedIncomeTaxesCurrent", "IncomeTaxExaminationPenaltiesAndInterestAccrued", "UnrecognizedTaxBenefitsIncomeTaxPenaltiesAndInterestAccrued","OtherAccruedLiabilitiesCurrent", "AccruedAdvertisingCurrent", "UnrecognizedTaxBenefitsIncomeTaxPenaltiesAccrued"], 6),
                    "NetCashGenerated": (["NetCashProvidedByUsedInOperatingActivities","NetCashProvidedByUsedInFinancingActivities","NetCashProvidedByUsedInInvestingActivities"], 3),
}
subtract_conversion = {
                    "Liabilities":["Assets","StockholdersEquity"],
                    "LiabilitiesNoncurrent": ["Liabilities","LiabilitiesCurrent"],
                    "LongTermDebtCurrent": ["LongTermDebt", "LongTermDebtNoncurrent"],
                    "AssetsNoncurrent": ["Assets", "AssetsCurrent"],
                    "CostOfRevenue":["Revenues","GrossProfit"],
                    "GrossProfit": ["Revenues","CostOfGoodsAndServicesSold"],
                    "OperatingIncomeLoss":["GrossProfit", "OperatingExpenses"],
                    "OperatingExpenses":["GrossProfit","OperatingIncomeLoss"],
                    "Depreciation":["DepreciationDepletionAndAmortization","AmortizationOfIntangibleAssets"],
                    "CostsAndExpenses":["Revenues","NetIncomeLoss"],
                    "CostOfGoodsAndServicesSold": ["CostsAndExpenses","OperatingExpenses"],
                    "InventoryNet": ["AssetsCurrent", "OtherThanInventoryCurrent"],
                    "FreeCashFlow": ["NetCashProvidedByUsedInOperatingActivities", "CapitalExpenditure"]
}

division_conversion = {
}

multiply_conversion = {
}

optional = ["DividendsPayableCurrent",
            "CapitalLeaseObligationsCurrent",
            "PensionAndOtherPostretirementDefinedBenefitPlansLiabilitiesNoncurrent",
            "CapitalLeaseObligationsNoncurrent",
            "DeferredRevenue",
            "AccountsReceivableNetNoncurrent",
            "PaymentsToAcquireOtherPropertyPlantAndEquipment",
            "CashAndCashEquivalentsAtCarryingValue",
            "AvailableForSaleSecuritiesCurrent",
            "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
            "BusinessCombinationRecognizedIdentifiableAssetsAcquiredAndLiabilitiesAssumedCurrentAssets",
            "PrepaidExpenseAndOtherAssetsCurrent",
            "DeferredTaxAssetsNetCurrent",
            "TradingSecuritiesCurrent",
            "DepositsAssetsCurrent",
            "AssetsHeldForSaleCurrent",
            "FinanceLeaseLiabilityPaymentsDueNextTwelveMonths",
            "FinanceLeaseLiabilityPaymentsRemainderOfFiscalYear",
            "LineOfCreditFacilityAmountOutstanding",
            "CommercialPaper",
            "LongTermDebtMaturitiesRepaymentsOfPrincipalAfterYearFive",
            "IncomeTaxExaminationPenaltiesAndInterestAccrued",
            "UnrecognizedTaxBenefitsIncomeTaxPenaltiesAndInterestAccrued",
            "OtherAccruedLiabilitiesCurrent",
            "AccruedAdvertisingCurrent",
            "UnrecognizedTaxBenefitsIncomeTaxPenaltiesAccrued"
            ]

shit_approximates = ["ApproxFreeCashFlow", "LongTermDebt", "AssetsNoncurrent", "Depreciation"]

dynamic_fuckers = ["ShortTermDebtWeightedAverageInterestRate"]

annual_measures = ["IncomeTaxesPaidNet", "DepreciationAndAmortization", "Depreciation", "AmortizationOfIntangibleAssets", "DepreciationDepletionAndAmortization"]

shares_ways = ["EntityCommonStockSharesOutstanding", "EntityPublicFloat", "WeightedAverageNumberOfSharesOutstandingBasic", "WeightedAverageNumberOfDilutedSharesOutstanding"]
#used in unitrun
valid_units = ['USD','shares','USD/shares', 'Year', 'Entity', 'Segment', 'USD/Contract', 'Job',  'pure', 'USD/Investment', 'Position']
all_units = ['Patent', 'USD', 'Restaurant', 'CNY', 'count', 'former_employee', 'State', 'state', 'membership', 'USD/rights', 'companies', 'Contracts', 'JPY/USD', 'EUR/shares', 'Cases', 'CHF/EUR', 'reportable_unit', 'businesses', 'stores', 'USD/warrant', 'employees', 'reportable_segments', 'derivative', 'Property', 'Employees', 'interest_rate_swap', 'USD/EUR', 'positions', 'Store', 'country', 'USD/Investment', 'CNY/shares', 'USD/shares_unit', 'Reporting_Unit', 'MXN/USD', 'item', 'day', 'uSDollarPerTonne', 'Rate', 'BRL', 'reportablesegments', 'LegalMatter', 'business_segment', 'Interest_Rate_Swap', 'JPY/EUR', 'plan', 'INR/shares', 'JPY', 'JPY/shares', 'numberofprojects', 'EUR', 'unit', 'Years', 'Job', 'years', 'USD/Decimal', 'instrument', 'GBP/EUR', 'reportable_segment', 'percentage', 'Contract', 'Plaintiff', 'TWD/shares', 'TWD', 'Account', 't', 'businessSegment', 'CHF', 'year', 'Positions', 'Projects', 'acquisitions', 'TWD/EUR', 'shares', 'GBP/shares', 'classAction', 'Interest_Rate_Swaps', 'reporting_unit', 'Investment', 'segement', 'Wells', 'segments', 'warehouse', 'AUD/EUR', 'Ground', 'project', 'Segments', 'reportableSegments', 'USD/Contract', 'GBP', 'Derivative', 'case_filed', 'BusinessSegment', 'DKK/shares', 'Acquisition', 'position', 'CNY/USD', 'location', 'defendant', 'operatingSegment', 'Year', 'Operating_segments', 'Y', 'company', 'Business', 'Tonne', 'plaintiff', 'businesscombinations', 'Derivatives', 'Reportable_Segments', 'HKD', 'MYR', 'units', 'operating_segments', 'employee', 'CHF/shares', 'EUR/USD', 'AED', 'patent', 'USD/Right', 'numberofyears', 'Segment', 'customer', 'subsidiaries', 'MXN/shares', 'legalmatter', 'NumberofBusinesses', 'acre', 'DKK', 'Option', 'sqft', 'Entity', 'Business_Segments', 'Employee', 'Person', 'claims', 'states', 'MXN', 'CAD/EUR', 'CAD', 'Location', 'account', 'reportableSegment', 'securities', 'Project', 'VEF/USD', 'business_unit', 'Country', 'Acquistions', 'Operating_Segment', 'Position', 'current_employees', 'Land', 'segment', 'CAD/shares', 'business', 'AUD', 'entity', 'acquisition', 'legal_action', 'countries', 'claim', 'D', 'Day', 'lawsuit', 'USD/PartnershipUnit', 'security', 'Percent', 'ZAR', 'INR/EUR', 'individuals', 'Stock_options', 'USD/shares', 'INR', 'pure', 'Businesses']