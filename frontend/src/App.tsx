import { Navigate, Route, Routes } from "react-router-dom";
import { RecordDetailsPage } from "./pages/RecordDetailsPage";
import { RecordsListPage } from "./pages/RecordsListPage";
import { ReviewQueuePage } from "./pages/ReviewQueuePage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<RecordsListPage />} />
      <Route path="/review-queue" element={<ReviewQueuePage />} />
      <Route path="/patients/:subjectId" element={<RecordDetailsPage />} />
      <Route path="/records/:subjectId/:hadmId" element={<RecordDetailsPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
