Prompt engineering and advanced prompting are powerful techniques that allow users to guide and shape the behavior of large language models (LLMs). By crafting well-designed prompts, business users and analysts in financial services can leverage the knowledge and capabilities of LLMs with enterprise-specific information to perform a wide range of tasks with remarkable effectiveness and efficiency. The strength of prompt engineering lies in its ability to distill complex queries or instructions into concise, yet expressive, prompts that can elicit relevant and coherent responses. This technique enables business users to leverage the model’s natural language understanding, reasoning and generation capabilities to tackle challenges ranging from text summarization, data analysis and interpretation, and financial calculations. Advanced prompting techniques, such as few-shot learning and chain-of-thought prompting, further enhance the models’ performance by providing exemplars or step-by-step reasoning processes, enabling them to exhibit human-like reasoning and problem-solving abilities. Prompt engineering and advanced prompting empower analysts to harness the full potential of LLMs and their hidden reasoning capabilities. Using Anthropic’s Claude 3 Sonnet model on Amazon Bedrock with financial data can enable financial analysts to provide contextual insights from various data modalities (image, text) combining with advanced prompting. It can help enhance analysts’ productivity through the ability to perform financial analysis and calculations using natural language (English) prompts, reducing time.

A research paper published by the University of Chicago Booth School of Business on “Financial Statement Analysis with Large Language Models” found the following –

“We investigate whether an LLM can successfully perform financial statement analysis in a way similar to a professional human analyst. We provide standardized and anonymous financial statements to GPT4 and instruct the model to analyze them to determine the direction of future earnings. Even without any narrative or industry-specific information, the LLM outperforms financial analysts in its ability to predict earnings changes…”

Financial analysis examines a company’s performance within the context of its financial statements (revenue, cash flow, asset, liabilities etc.). In section 1, we show how financial analysts can analyze financial statements (the balance sheet, income statement, and cash flow statement) using Generative AI, Anthropic’s Claude 3 Sonnet on Amazon Bedrock, and prompt engineering. Capital markets customers have access to information about macroeconomic events and index price movements, which can be leveraged by research analysts and quantitative analysts to study the relationship between these events and their impact on index prices. In section 2, we show how Anthropic’s Claude 3 Sonnet on Amazon Bedrock can analyze the impact of macroeconomic events on index prices, incorporating multi-modal data (image and text) with macroeconomic event information, to gain insights like impact of inflation or geopolitics on index price action.

Section 1: Financial statement analysis with generative AI and LLM
Financial analysis centers on evaluating a company’s ability to earn a return on its capital that meets or exceeds the cost of that capital, profitably grows its operations, and generates sufficient cash to meet obligations and pursue opportunities. This analysis begins with the information found in a company’s financial reports, including audited financial statements, additional disclosures required by regulatory authorities, and any accompanying (unaudited) management commentary. Basic financial statement analysis, as presented in this context, provides a foundation that enables the analyst to better understand and interpret additional information gathered from research beyond the financial reports.

The three main types of financial statements that financial analysts work with are the balance sheet, income statement, and cash flow statement. The income statement, also known as the profit and loss statement, reports a company’s revenues, expenses, and net income or loss over a specific period, typically a quarter or a year. It shows how much money the company made or lost during that time frame. To interpret the income statement, analysts analyze the revenue growth, expense management, and profitability trends. The balance sheet presents a snapshot of a company’s assets, liabilities, and shareholders’ equity at a specific point in time. It shows what the company owns (assets), what it owes (liabilities), and the residual interest of shareholders (equity). Using the balance sheet, analysts evaluate the company’s liquidity (ability to meet short-term obligations), solvency (ability to meet long-term obligations), and financial leverage (ratio of debt to equity). The cash flow statement reports the inflows and outflows of cash during a specific period, categorized into three sections – operating activities, investing activities and financing activities. Financial analysts assess the company’s ability to generate positive cash flow from operations, its investment needs, and its financing activities. These financial statements are interconnected and should be analyzed together to gain a comprehensive understanding of a company’s financial performance, position, and liquidity.

Benefits of financial statement analysis with LLM and Generative AI (Anthropic Claude 3 Sonnet on Amazon BedrockEmpowering analysts to perform financial statement analysis image 1
The following shows financial statement analysis for investment research with Anthropic Claude 3 Sonnet on Amazon Bedrock using image, text data, and prompts. Watch the video about financial statement analysis for investment research with Anthropic Claude 3 Sonnet on Amazon Bedrock using image, text data, and prompts.
Video screen capture
Example financial statement analysis prompts and results

Financial analysis prompt questions 1:
How much debt does the company have?

LLM Results:

LLM results

Financial analysis prompt questions 2:
Are revenues steadily increasing over time? Give the response in bullet points

LLM Results:

LLM results 2Financial analysis prompt questions 3:

Perform a financial analysis by calculating the ratios from the data in the images. Interpret the ratios. Give your answer in Tabular format indifferent section with 3 columns – Ratios for that section, Ratio calculation, Interpretation for the ratio values. For every ratio, give calculations/basis/justifications. Do not make up answers/numbers/data. Answer only if you know it.

Activity Ratios:

| Activity Ratios | Ratio calculation |
|—————–|——————-|
| Inventory turnover | Cost of goods sold / Average inventory |
| Days of inventory on hand (DOH) | Number of days in period / Inventory turnover |
| Receivables turnover | Revenue or Net credit sales / Average receivables |
| Days of sales outstanding (DSO) | Number of days in period / Receivables turnover |
| Payable Turnover | Purchases / Average payables |
| Number of days of payables | Number of days in period / Payable turnover |
| Working capital turnover | Revenue / Average working capital |
| Fixed assets turnover | Revenue / Average fixed assets |
| Total assets turnover | Revenue / Average total assets |

LLM Results:LLM results 3

Financial analysis prompt questions 4:

Perform a financial analysis by calculating the ratios from the data in the images. Interpret the ratios. Give your answer in Tabular format indifferent section with 3 columns – Ratios for that section, Ratio calculation, Interpretation for the ratio values. For every ratio, give calculations/basis/justifications. Do not make up answers/numbers/data. Answer only if you know it.

Profitability Ratios:

| Return on sales ratios | Ratio calculation |
|————————|——————-|
| Gross profit margin | Gross profit / Revenue |
| Operating margin | Operating profit / Revenue |
| Pretax margin | EBT (Earnings before taxes) / Revenue |
| Net profit margin | Net income / Revenue |

LLM Results:

LLM results 4

Section 2: Hypothesis testing and cause-effect analysis for investment research with generative AI and LLM
There are different factors that affect global capital markets. Capital markets fluctuate across a range of macroeconomic indicators that provide insights into the overall health of the economy. Factors such as Gross Domestic Product (GDP) growth, unemployment rates, inflation, consumer confidence, and the Federal Reserve’s interest rate decisions can impact investor sentiment and market behavior. Strong GDP growth and low unemployment typically signal a robust economy; leading to increased investor confidence and higher stock prices. High inflation coupled with rising interest rates can dampen investor sentiment; resulting in market corrections or declines as borrowing costs increase and consumer spending decreases.

The Federal Reserve monitors key inflation indicators, including the Personal Consumption Expenditures (PCE), Producer Price Index (PPI), and Consumer Price Index (CPI), to guide its monetary policy decisions. When these metrics exceed expectations, the Fed may raise interest rates to curb inflationary pressures. Higher interest rates can impact corporate profits and consumer spending, influencing capital markets.

Global events, such as geopolitical conflicts, pandemics, and wars, can also affect capital markets to varying degrees. Investors respond negatively to unexpected adverse events, leading to increased market volatility and economic uncertainty. Capital markets also identify and allocate capital towards emerging macro themes, such as artificial intelligence, sustainability, and the internet of things. Investors seek to generate returns as these themes are executed and translated into new products and services.

The Federal Reserve and the Securities and Exchange Commission (SEC) play crucial roles in regulating and overseeing the capital markets in the United States. The Fed conducts monetary policy to promote economic growth, price stability, and maximum employment, while the SEC oversees the securities markets, protects investors, and enforces compliance with securities laws.

By understanding the interplay between macroeconomic indicators, global events, and regulatory oversight, capital market participants can make informed investment decisions and navigate the dynamic financial landscape. Investment research is the cornerstone of successful investing and involves gathering and analyzing relevant information about potential investment opportunities. Through thorough research, research analysts, and quants come up with a hypothesis, test the hypothesis with data, and understand the cause-effect of different events on price movements of leading indexes like S&P and Dow Jones, before portfolio managers decide, allocate capital to strategies and mitigate risks.

The following shows hypothesis testing and cause-effect analysis for investment research with Anthropic Claude 3 Sonnet on Amazon Bedrock using image, text data, and prompts. Watch a video about hypothesis testing and cause-effect analysis for investment research with Anthropic Claude 3 Sonnet on Amazon Bedrock using image, text data, and prompts.
Anthropic Claude 3 Sonnet on Amazon Bedrock screen capture

Example hypothesis testing prompts and results
Hypothesis testing prompts 1: Build a pivot table out of the generated table. For the pivot table, rows will be category and columns will be price movement and values will be the count of price movement. Create a new column in the pivot table with total for each row. Create a grand total for each column.

Results:

Pivot table

InterpretationHypothesis testing prompts 2:  Identify what technological innovations have a positive impact on price action? Give the response in a table.

LLM Results:

table listingHypothesis testing prompts 3:  Identify what economic events have a positive impact on price action? Give the response in a table.

LLM Results:

table listing 2

Dive deeper into the solution
To dive deeper into the solution and the code shown in this post, check out the GitHub repo. The code base gives step-by-step guidance on deploying and using the solutions with UI for financial statement analysis, and for hypothesis testing and cause-effect analysis.

Following are the three components of invoking Amazon Bedrock API for Anthropic Claude 3 Sonnet model using both image data and text prompts.

Anthropic Claude 3 Messages API format

messages = []
    
    for chat_msg in chat_messages:
        if (chat_msg.message_type == 'image'):
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": chat_msg.text,
                        },
                    }
                ]
            })
        else:
            messages.append({
                "role": chat_msg.role,
                "content": [
                    {
                        "type": "text",
                        "text": chat_msg.text
                    }
                ]
            })
            
    return messages
Body builder function for Amazon Bedrock

response = bedrock.invoke_model(body=body, modelId="anthropic.claude-3-sonnet-20240229-v1:0", contentType="application/json", accept="application/json")
    
response_body = json.loads(response.get('body').read()) # read the response
    
output = response_body['content'][0]['text']
    
response_message = ChatMessage('assistant', 'text', output)

Conclusion
Anthropic’s Claude 3 Sonnet model on Amazon Bedrock provides analysts with powerful capabilities to enhance their productivity and drive more insightful analysis. By leveraging the model’s multimodal abilities to analyze text, images, and quantitative data, analysts can test hypotheses, uncover cause-and-effect relationships, and gain a deeper understanding of how macroeconomic events impact capital markets. Anthropic Claude 3 Sonnet on Amazon Bedrock can be used for investment research, integrating diverse data sources like news articles, reports, and economic indicators to analyze the effects of major events on index prices. This technology speeds up the research process and generates valuable insights to inform investment strategies. The model’s natural language processing capabilities make it well-suited for financial statement analysis. By ingesting income statements, balance sheets, and cash flow statements, Anthropic Claude 3 Sonnet on Amazon Bedrock interprets the data, identifies key financial ratios and trends, and provides analysts with a comprehensive assessment of a company’s financial health and performance. As the financial services industry continues to grapple with an ever-increasing volume and variety of data, solutions like Anthropic’s Claude 3 Sonnet model on Amazon bedrock offer a way to streamline analysis, uncover hidden insights, and drive more informed decision-making. To experience the power of Anthropic’s Claude 3 Sonnet model on Amazon Bedrock, analysts are encouraged to explore the platform’s capabilities and leverage its advanced features to enhance their analysis and decision-making processes using their own data.