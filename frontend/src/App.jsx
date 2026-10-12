import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [risk, setRisk] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [activeTab, setActiveTab] = useState("new");

  const askQuestion = async () => {
  if (!question.trim()) {
    setError("Please enter a question.");
    return;
  }

  setLoading(true);
  setError("");
  setAnswer("");
  setSources([]);
  setRisk(null);

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question.trim(),
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Request failed.");
    }

    // =========================================================
    // API RESULT
    // =========================================================

    const result = data.result || {};

    // =========================================================
    // ANSWER
    // =========================================================

    const finalAnswer =
      result.answer ||
      data.answer ||
      result.final_answer ||
      "The RAG system completed successfully.";

    setAnswer(finalAnswer);

    // =========================================================
    // RISK INFORMATION
    // =========================================================
    // Backend returns:
    //
    // risk: {
    //   level: "LOW",
    //   score: 0.2,
    //   reasoning: "..."
    // }
    //
    // Frontend expects:
    //
    // risk_level
    // risk_score
    // reasoning
    //
    // So we normalize the backend response here.

    if (result.risk) {
      setRisk({
        risk_level:
          result.risk.level ||
          result.risk.risk_level ||
          "UNKNOWN",

        risk_score:
          result.risk.score !== undefined
            ? result.risk.score
            : result.risk.risk_score !== undefined
            ? result.risk.risk_score
            : 0,

        reasoning:
          result.risk.reasoning ||
          "Risk information available from the RAG pipeline.",
      });
    } else if (
      result.risk_level ||
      result.risk_score !== undefined
    ) {
      setRisk({
        risk_level: result.risk_level || "UNKNOWN",

        risk_score:
          result.risk_score !== undefined
            ? result.risk_score
            : 0,

        reasoning:
          result.risk_reasoning ||
          result.reasoning ||
          "Risk information available from the RAG pipeline.",
      });
    }

    // =========================================================
    // RETRIEVED SOURCES
    // =========================================================

    const retrievedSources =
      result.sources ||
      result.retrieved_documents ||
      result.documents ||
      [];

    if (Array.isArray(retrievedSources)) {
      setSources(retrievedSources);
    }

    // =========================================================
    // QUERY HISTORY
    // =========================================================

    setHistory((previous) => [
      {
        question: question.trim(),
        answer: finalAnswer,
        time: new Date().toLocaleTimeString(),
      },
      ...previous,
    ]);

  } catch (err) {
    console.error(err);

    setError(
      err.message ||
      "Unable to connect to the RAG API."
    );

  } finally {
    setLoading(false);
  }
};

  const clearQuestion = () => {
    setQuestion("");
    setAnswer("");
    setSources([]);
    setRisk(null);
    setError("");
  };

  const formatSource = (source, index) => {
    if (typeof source === "string") {
      return source;
    }

    if (source?.text) {
      return source.text;
    }

    if (source?.content) {
      return source.content;
    }

    if (source?.document) {
      return source.document;
    }

    return `Retrieved document ${index + 1}`;
  };

  return (
    <div className="app">

      {/* ================================================= */}
      {/* HEADER */}
      {/* ================================================= */}

      <header className="header">
        <div className="brand">
          <div className="brand-icon">✦</div>

          <div>
            <h1>AI Support Assistant</h1>
            <p>Internal knowledge base powered by RAG technology</p>
          </div>
        </div>
      </header>


      {/* ================================================= */}
      {/* STAT CARDS */}
      {/* ================================================= */}

      <section className="stats">

        <div className="stat-card">
          <div className="stat-icon blue">▢</div>
          <div>
            <span>Queries Today</span>
            <strong>{history.length}</strong>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon green">◷</div>
          <div>
            <span>Avg Response</span>
            <strong>{loading ? "..." : "—"}</strong>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon purple">✓</div>
          <div>
            <span>Accuracy</span>
            <strong>—</strong>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon orange">▤</div>
          <div>
            <span>Documents</span>
            <strong>—</strong>
          </div>
        </div>

      </section>


      {/* ================================================= */}
      {/* MAIN CONTENT */}
      {/* ================================================= */}

      <main className="main-grid">

        {/* LEFT PANEL */}

        <section className="query-panel">

          <div className="tabs">

            <button
              className={activeTab === "new" ? "tab active" : "tab"}
              onClick={() => setActiveTab("new")}
            >
              New Query
            </button>

            <button
              className={activeTab === "history" ? "tab active" : "tab"}
              onClick={() => setActiveTab("history")}
            >
              Query History
            </button>

          </div>


          {/* ================================================= */}
          {/* NEW QUERY */}
          {/* ================================================= */}

          {activeTab === "new" && (
            <div className="query-content">

              <label>
                Enter customer question or issue
              </label>

              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="e.g., What are the four verticals of PMAY-U 2.0?"
              />


              <div className="action-row">

                <button className="secondary-btn">
                  ▧ Upload Screenshot
                </button>

                <button className="secondary-btn">
                  ↑ Attach File
                </button>

              </div>


              <button
                className="generate-btn"
                onClick={askQuestion}
                disabled={loading}
              >
                {loading
                  ? "⟳ Generating Response..."
                  : "✦ Generate AI Response  ➤"}
              </button>


              {/* ERROR */}

              {error && (
                <div className="error-box">
                  <strong>Request failed</strong>
                  <p>{error}</p>
                </div>
              )}


              {/* ANSWER */}

              {answer && (
                <div className="answer-box">

                  <div className="answer-header">
                    <span className="answer-icon">✦</span>
                    <h2>AI Response</h2>
                  </div>

                  <div className="answer-content">
                    {answer}
                  </div>

                </div>
              )}

            </div>
          )}


          {/* ================================================= */}
          {/* HISTORY */}
          {/* ================================================= */}

          {activeTab === "history" && (
            <div className="history-content">

              {history.length === 0 ? (
                <div className="empty-history">
                  <div>◷</div>
                  <p>No queries yet.</p>
                </div>
              ) : (
                history.map((item, index) => (
                  <div className="history-item" key={index}>
                    <div>
                      <strong>{item.question}</strong>
                      <small>{item.time}</small>
                    </div>

                    <p>{item.answer}</p>
                  </div>
                ))
              )}

            </div>
          )}

        </section>


        {/* ================================================= */}
        {/* RIGHT SIDEBAR */}
        {/* ================================================= */}

        <aside className="sidebar">


          {/* ================================================= */}
{/* KNOWLEDGE SOURCES */}
{/* ================================================= */}

<div className="side-card">

  <div className="side-title">
    <span>▤</span>
    <h3>Knowledge Sources</h3>
  </div>

  {sources.length === 0 ? (

    <div className="empty-card">

      <div className="empty-icon">
        !
      </div>

      <p>
        Sources will appear here after your query.
      </p>

    </div>

  ) : (

    <div className="sources-list">

      {sources.map((source, index) => (

        <details
          className="source-item"
          key={index}
        >

          <summary className="source-summary">

            <span className="source-number">
              {index + 1}
            </span>

            <div className="source-main">

              <strong>
                {source.source || `Retrieved document ${index + 1}`}
              </strong>

              <span>
                Page: {source.page || "Unknown"}
              </span>

            </div>

            <span className="source-arrow">
              ▾
            </span>

          </summary>


          <div className="source-details">

            <div className="source-meta">

              <span>
                <strong>Source:</strong>{" "}
                {source.source || "Unknown"}
              </span>

              <span>
                <strong>Page:</strong>{" "}
                {source.page || "Unknown"}
              </span>

            </div>


            {source.distance !== null &&
             source.distance !== undefined && (

              <div className="source-score">

                Retrieval Distance:{" "}

                {typeof source.distance === "number"
                  ? source.distance.toFixed(4)
                  : source.distance}

              </div>

            )}


            {source.chunk_id &&
             source.chunk_id !== "Unknown" && (

              <div className="source-chunk">

                <strong>Chunk ID:</strong>{" "}

                {source.chunk_id}

              </div>

            )}


            <div className="source-text">

              <strong>Retrieved Evidence</strong>

              <p>
                {source.text ||
                  "No text available for this source."}
              </p>

            </div>

          </div>

        </details>

      ))}

    </div>

  )}

</div>


          {/* ================================================= */}
          {/* RISK DETECTION */}
          {/* ================================================= */}

          <div className="side-card">

            <div className="side-title">
              <span>⚠</span>
              <h3>Risk Detection</h3>
            </div>

            {risk ? (

              <div className="risk-content">

                <div className="risk-row">
                  <span>Risk Level</span>
                  <strong className="risk-value">
                    {risk.risk_level || "—"}
                  </strong>
                </div>

                <div className="risk-row">
                  <span>Risk Score</span>
                  <strong>
                    {risk.risk_score !== undefined
                      ? risk.risk_score
                      : "—"}
                  </strong>
                </div>

                <div className="risk-reasoning">
                  <span>Reasoning</span>
                  <p>
                    {risk.reasoning ||
                      "Risk information available from the RAG pipeline."}
                  </p>
                </div>

              </div>

            ) : (

              <div className="empty-card">

                <div className="empty-icon">!</div>

                <p>
                  Risk information will appear after your query.
                </p>

              </div>

            )}

          </div>


          {/* ================================================= */}
          {/* LANGUAGE DETECTION */}
          {/* ================================================= */}

          <div className="side-card">

            <div className="side-title">
              <span>文</span>
              <h3>Language Detection</h3>
            </div>

            <div className="language-row">
              <span>Current Language</span>
              <strong>English</strong>
            </div>

            <div className="language-row">
              <span>Auto-detect</span>
              <strong className="enabled">
                Enabled
              </strong>
            </div>

            <div className="supported">
              Supported: English, German, French, Spanish
            </div>

          </div>


        </aside>

      </main>

    </div>
  );
}

export default App;