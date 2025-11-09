import React, { useState } from "react";
import { Search, Loader2, ExternalLink, FileText, AlertCircle } from "lucide-react";

export default function App() {
  const [query, setQuery] = useState("");
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [processingTime, setProcessingTime] = useState(0);

  // Backend API URL
  const API_BASE_URL = "http://127.0.0.1:8000";

  const sampleQueries = [
    "I am hiring for Java developers who can also collaborate effectively with my business teams.",
    "Looking to hire mid-level professionals who are proficient in Python, SQL, and JavaScript.",
    "Need assessments for analyst role with cognitive and personality tests.",
  ];

  const handleRecommend = async () => {
    if (!query.trim()) {
      setError("Please enter a query or job description.");
      return;
    }

    setLoading(true);
    setError("");
    setRecommendations([]);
    const startTime = Date.now();

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: query.trim() }),
      });

      if (!response.ok) throw new Error(`API Error: ${response.status}`);

      const data = await response.json();
      setRecommendations(data.recommendations || []);
      setProcessingTime(Date.now() - startTime);
    } catch (err) {
      setError(
        err.message || "Failed to fetch recommendations. Please check your API connection."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <div className="max-w-6xl mx-auto p-6">
        {/* Header */}
        <div className="text-center mb-8 pt-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            SHL Assessment Recommender
          </h1>
          <p className="text-gray-600 text-lg">
            Assessment recommendations for hiring needs
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-xl shadow-md p-6 mb-6">
          <label className="block text-sm font-semibold text-gray-700 mb-2">
            Enter Job Description or Query
          </label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Example: Need a developer skilled in Java and communication."
            className="w-full p-4 border border-gray-300 rounded-lg focus:border-blue-500 focus:ring-2 focus:ring-blue-200 outline-none resize-none h-32 text-gray-700"
          />

          <div className="mt-4">
            <p className="text-sm font-medium text-gray-600 mb-2">Try a sample query:</p>
            <div className="flex flex-wrap gap-2">
              {sampleQueries.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sample)}
                  className="text-xs bg-blue-50 hover:bg-blue-100 text-blue-700 px-3 py-2 rounded-lg transition-colors"
                >
                  {sample.substring(0, 50)}...
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleRecommend}
            disabled={loading}
            className="mt-6 w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-semibold py-3 rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                Analyzing...
              </>
            ) : (
              <>
                <Search size={20} />
                Get Recommendations
              </>
            )}
          </button>
        </div>

        {/* Error Message */}
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-lg flex items-start gap-3">
            <AlertCircle className="text-red-500 flex-shrink-0 mt-0.5" size={20} />
            <div>
              <p className="font-semibold text-red-800">Error</p>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Recommendations Table */}
        {recommendations.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Recommended Assessments ({recommendations.length})
              </h2>
              <span className="text-sm text-gray-500">
                Processed in {processingTime}ms
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full border-collapse">
                <thead>
                  <tr className="border-b-2 border-gray-200 bg-gray-50">
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">#</th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">
                      Assessment Name
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">
                      Description
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">
                      Score
                    </th>
                    <th className="text-left py-3 px-4 font-semibold text-gray-700">Link</th>
                  </tr>
                </thead>
                <tbody>
                  {recommendations.map((rec, idx) => (
                    <tr
                      key={idx}
                      className="border-b border-gray-100 hover:bg-blue-50 transition-colors"
                    >
                      <td className="py-3 px-4 text-gray-600 font-medium">{idx + 1}</td>
                      <td className="py-3 px-4 font-semibold text-gray-900">
                        {rec.assessment_name}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600 max-w-md truncate">
                        {rec.description || "No description"}
                      </td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                          {(rec.relevance_score * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        <a
                          href={rec.assessment_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 font-medium"
                        >
                          <ExternalLink size={16} />
                          View
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Placeholder */}
        {!loading && recommendations.length === 0 && !error && (
          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-8 text-center">
            <FileText className="mx-auto mb-4 text-blue-500" size={48} />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">
              Ready to Find the Perfect Assessments
            </h3>
            <p className="text-gray-600">
              Enter a job description or requirements above to get assessment
              recommendations.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

