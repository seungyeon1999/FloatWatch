"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ArrowUp, Bot, ChevronRight, Send, X } from "lucide-react";
import { api } from "@/lib/api";
import type { ContentItem } from "@/lib/types";

type ChatMessage = { id: number; role: "bot" | "user"; text: string; action?: { label: string; view: string }; topic?: string };
type ChatGuide = { words: string[]; text: string; action?: { label: string; view: string }; topic?: string };

const quickQuestions = ["이미지 탐색은 어떻게 해요?", "오늘 뭐 물어볼까요?", "프로젝트 개요는 뭐예요?", "실시간 탐색은 어떻게 해요?"];
const followUpsByView: Record<string, string[]> = {
  overview: ["프로젝트 개요는 뭐예요?", "무엇을 탐지할 수 있나요?", "부유물 탐색은 어떻게 해요?", "결과는 어디에 저장되나요?"],
  development: ["개발정보는 뭘 보여줘요?", "어떤 PT 파일을 지원하나요?", "분석 시간은 얼마나 걸리나요?", "Detection과 Segmentation 차이는?"],
  analysis: ["이미지 탐색은 어떻게 해요?", "동영상 탐색은 어떻게 해요?", "어떤 PT 파일을 지원하나요?", "결과는 어디에 저장되나요?"],
  realtime: ["실시간 탐색은 어떻게 해요?", "카메라 연결은 어떻게 해요?", "실시간 탐지 기록은 어디서 봐요?", "어떤 모델을 선택하나요?"],
  records: ["탐색 기록은 어디서 봐요?", "두 모델은 어떻게 비교하나요?", "신뢰도는 정확도인가요?", "분석 실패 시 어떻게 하나요?"],
  inquiry: ["1:1 문의는 어떻게 해요?", "분석 실패 시 어떻게 하나요?", "업로드 용량 제한은?", "내 기록은 다른 사람도 보나요?"],
  login: ["로그인은 어떻게 해요?", "내 기록은 다른 사람도 보나요?", "이미지 탐색은 어떻게 해요?", "1:1 문의는 비공개인가요?"],
  home: ["메인페이지 기능 알려줘", "이미지 탐색은 어떻게 해요?", "실시간 탐색은 어떻게 해요?", "탐색 기록은 어디서 봐요?"],
  chat: ["오늘 뭐 물어볼까요?", "FloatWatch 쉽게 설명해줘", "분석 시작 도와줘", "짧게 응원해줘"],
};
const guides: ChatGuide[] = [
  {
    words: ["메인페이지", "메인 페이지", "홈 화면", "첫 화면", "기능", "메뉴", "무슨 기능", "어떤 기능"],
    text: "메인페이지에서는 프로젝트 개요와 개발정보를 확인하고, 로그인 후 부유물 탐색, 실시간 탐색, 탐색 기록, 1:1 문의로 이동할 수 있습니다. 파일 분석은 부유물 탐색에서, 웹캠 기반 확인은 실시간 탐색에서 시작하면 됩니다.",
    action: { label: "홈으로 이동", view: "home" },
  },
  {
    words: ["프로젝트 개요", "프로젝트 목적", "만든 목적", "만든 이유", "개발 목적", "개발 이유", "개발 배경", "왜 만들", "무엇을 위해"],
    text: "프로젝트 개요는 FloatWatch가 어떤 흐름으로 바다를 관측하는지 보여주는 화면입니다. 영상 입력, AI 분석, 결과 확인까지 이어지는 처리 흐름과 탐지, 이해, 대응이라는 핵심 가치를 확인할 수 있습니다.",
    action: { label: "프로젝트 개요 보기", view: "overview" },
  },
  {
    words: ["심심", "놀아", "대화", "잡담", "얘기", "말동무", "뭐하지", "뭐 할까"],
    text: "좋아요, 잠깐 같이 얘기해요. 가볍게 수다를 떨어도 되고, FloatWatch 기능을 하나씩 둘러봐도 좋아요. 지금은 이미지 탐색, 실시간 탐색, 프로젝트 개요 같은 걸 물어보면 바로 안내해드릴 수 있습니다.",
    topic: "chat",
  },
  {
    words: ["안녕", "반가워", "하이", "hello", "hi", "ㅎㅇ"],
    text: "안녕하세요. 오늘은 FloatWatch 기능을 같이 살펴봐도 좋고, 그냥 편하게 질문을 던져도 괜찮아요. 이미지 탐색이나 실시간 탐색부터 물어보면 바로 이어서 도와드릴게요.",
    topic: "chat",
  },
  {
    words: ["뭐해", "넌 뭐해", "뭐 할 수", "할 수 있어", "도와줄 수", "무엇을 할 수"],
    text: "저는 FloatWatch 사용법을 안내하고, 기능을 쉽게 풀어 설명하고, 막히는 상황을 같이 정리할 수 있어요. 프로젝트 개요, 개발정보, 부유물 탐색, 실시간 탐색, 탐색 기록, 1:1 문의까지 물어보면 짧게 답해드릴게요.",
    topic: "chat",
  },
  {
    words: ["기분", "힘들", "어려워", "막막", "귀찮", "피곤", "졸려", "답답"],
    text: "그럴 때 있죠. 너무 한 번에 다 보려고 하지 말고, 지금 필요한 것 하나만 잡아도 충분해요. 예를 들면 '이미지 탐색 어떻게 해?'처럼 물어보면 제가 단계별로 짧게 정리해드릴게요.",
    topic: "chat",
  },
  {
    words: ["추천", "뭐부터", "처음", "처음이면", "먼저", "순서", "시작할까"],
    text: "처음이라면 프로젝트 개요로 전체 흐름을 보고, 개발정보에서 MVP 구성을 확인한 뒤, 부유물 탐색에서 이미지나 동영상을 분석해보는 순서가 좋습니다. 웹캠 테스트가 필요하면 실시간 탐색으로 가면 됩니다.",
    action: { label: "프로젝트 개요 보기", view: "overview" },
  },
  {
    words: ["쉽게 설명", "쉽게 말해", "간단히", "짧게", "요약", "한 줄"],
    text: "FloatWatch는 바다나 하천 영상에서 떠다니는 물체를 AI 모델로 찾아내고, 결과를 기록으로 정리해주는 서비스입니다. 이미지나 동영상을 넣으면 탐지 결과와 통계를 확인할 수 있어요.",
    action: { label: "프로젝트 개요 보기", view: "overview" },
  },
  {
    words: ["고마워", "감사", "좋아", "도움", "수고"],
    text: "천만에요. 필요할 때 바로 옆에서 FloatWatch 이용 방법을 짧고 정확하게 안내해드릴게요.",
    topic: "chat",
  },
  {
    words: ["응원", "칭찬", "할 수 있을까", "잘 될까", "파이팅", "화이팅"],
    text: "충분히 할 수 있어요. 지금처럼 하나씩 확인하면서 고치면 됩니다. 막히는 부분을 그대로 적어주면 제가 원인부터 다음 행동까지 같이 정리해드릴게요.",
    topic: "chat",
  },
  {
    words: ["농담", "웃긴", "재밌는 말", "웃겨"],
    text: "좋아요. FloatWatch식 농담으로 가면, 부유물은 떠다니지만 오류는 떠다니게 두면 안 됩니다. 발견하면 바로 잡아야죠. 이제 어떤 기능을 같이 볼까요?",
    topic: "chat",
  },
  {
    words: ["개발정보", "개발 정보", "개발", "기술", "스택", "구현", "mvp", "구성", "어떻게 만들"],
    text: "개발정보에서는 FloatWatch의 MVP 구성을 확인할 수 있습니다. YOLOv8·YOLO11 PT 모델과 분석 미디어를 입력하고, CPU 기반 추론을 실행한 뒤 클래스 통계, 처리 성능, 결과 기록을 확인하는 구조입니다.",
    action: { label: "개발정보 보기", view: "development" },
  },
  {
    words: ["실시간 탐색", "실시간", "웹캠", "카메라", "라이브", "cctv", "드론", "연안 카메라"],
    text: "실시간 탐색은 웹캠을 연결해 화면을 보면서 AI 탐지 결과를 확인하는 기능입니다. 모델을 등록한 뒤 실시간 탐색에서 카메라를 연결하고 촬영 위치를 선택한 다음 탐지 시작을 누르면 최근 탐지 이벤트가 기록됩니다.",
    action: { label: "실시간 탐색 열기", view: "realtime" },
  },
  {
    words: ["카메라 연결", "웹캠 연결", "카메라 권한", "권한", "촬영 위치", "위치 선택"],
    text: "실시간 탐색에서 카메라 연결을 누르고 브라우저 권한을 허용하세요. 탐지 시작 전에 촬영 위치를 선택해야 세션 기록이 저장되며, 실행 중에는 일시정지하거나 카메라를 종료할 수 있습니다.",
    action: { label: "실시간 탐색 열기", view: "realtime" },
  },
  {
    words: ["학습", "라벨링", "로보플로우", "roboflow", "데이터셋", "훈련"],
    text: "FloatWatch는 AI 모델을 직접 학습시키지 않습니다. Roboflow 등에서 라벨링과 학습을 완료한 YOLO PT 모델을 업로드해 이미지나 동영상에서 추론하고 결과를 확인하는 서비스입니다.",
    action: { label: "개발정보 보기", view: "development" },
  },
  {
    words: ["지원 형식", "파일 형식", "확장자", "이미지 형식", "동영상 형식", "mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png", "webp", "bmp"],
    text: "미디어는 JPG·JPEG·PNG·WEBP·BMP 이미지와 MP4·AVI·MOV·MKV·WEBM 동영상을 지원합니다. 미디어는 최대 2GB까지 업로드할 수 있습니다.",
    action: { label: "미디어 업로드", view: "analysis" },
  },
  {
    words: ["용량", "최대 크기", "몇 기가", "몇기가", "업로드 제한", "파일 크기"],
    text: "AI 모델은 .pt 형식으로 최대 500MB, 이미지와 동영상은 최대 2GB까지 업로드할 수 있습니다. 게시글과 1:1 문의 첨부파일은 파일당 최대 20MB입니다.",
    action: { label: "부유물 탐색 열기", view: "analysis" },
  },
  {
    words: ["박스", "바운딩", "세그먼트", "세그멘트", "segmentation", "detection", "마스크"],
    text: "YOLO Detection 모델은 부유물 위치를 바운딩 박스로 표시하고, Segmentation 모델은 물체 영역을 마스크 형태로 표시합니다. 업로드한 PT 모델의 작업 유형에 따라 결과 표현이 달라집니다.",
    action: { label: "부유물 탐색 열기", view: "analysis" },
  },
  {
    words: ["무엇을 탐지", "탐지 대상", "어떤 쓰레기", "쓰레기 종류", "클래스", "부유물 종류"],
    text: "탐지 대상과 클래스 이름은 업로드한 PT 모델의 학습 내용에 따라 결정됩니다. 폐플라스틱, 부표, 나무 조각 등 원하는 부유물 클래스로 학습한 모델을 사용하면 해당 종류와 위치를 구분해 표시할 수 있습니다.",
    action: { label: "프로젝트 개요 보기", view: "overview" },
  },
  {
    words: ["정확도", "map", "성능 지표", "평가 지표", "신뢰도", "처리 속도", "fps"],
    text: "분석 결과에서는 탐지 개수, 평균 신뢰도, 클래스별 통계와 처리 FPS를 확인할 수 있습니다. 이는 업로드한 미디어에서의 추론 결과이며, 정답 라벨 기반의 mAP처럼 모델의 공식 검증 정확도를 대신하지는 않습니다.",
    action: { label: "탐색 기록 보기", view: "records" },
  },
  {
    words: ["신뢰도 기준", "최소 신뢰도", "confidence", "오탐", "미탐"],
    text: "최소 신뢰도는 결과에 포함할 탐지의 기준값입니다. 값을 높이면 확실한 탐지만 남아 오탐이 줄 수 있지만 미탐이 늘 수 있고, 낮추면 더 많은 후보를 확인할 수 있습니다.",
    action: { label: "부유물 탐색 열기", view: "analysis" },
  },
  {
    words: ["gpu", "그래픽카드", "cpu", "분석 시간", "얼마나 걸", "처리 시간", "속도"],
    text: "현재 MVP는 별도 GPU 없이 로컬 PC의 CPU로 추론합니다. 처리 시간은 영상 길이와 해상도, 모델 크기, PC 성능에 따라 달라지므로 긴 영상은 시간이 더 걸릴 수 있습니다.",
    action: { label: "개발정보 보기", view: "development" },
  },
  {
    words: ["로그인", "회원가입", "카카오", "네이버", "구글", "소셜"],
    text: "이메일 계정으로 가입하거나 카카오·네이버·Google 간편 로그인을 이용할 수 있습니다. 부유물 탐색, 탐색 기록과 1:1 문의는 로그인 후 사용할 수 있습니다.",
    action: { label: "로그인 화면 열기", view: "login" },
  },
  {
    words: ["다른 사용자", "남이", "공개", "보이나", "개인정보", "내 모델", "내 영상", "보안"],
    text: "업로드한 미디어와 PT 모델, 분석 기록은 사용자 계정별로 구분됩니다. 다른 일반 사용자의 목록에는 노출되지 않으며, 1:1 문의도 작성자와 관리자만 확인할 수 있습니다.",
    action: { label: "마이페이지로 이동", view: "home" },
  },
  {
    words: ["실패", "오류", "안돼", "안되", "멈춤", "업로드 문제", "분석 문제"],
    text: "먼저 파일 형식과 용량, PT 모델이 YOLOv8 또는 YOLO11 기반인지 확인해주세요. 계속 실패하면 오류가 발생한 단계와 파일 정보를 적어 1:1 문의로 남겨주시면 확인할 수 있습니다.",
    action: { label: "1:1 문의하기", view: "inquiry" },
  },
  {
    words: ["이미지 탐색", "이미지 분석", "사진 탐색", "사진 분석", "이미지 업로드", "사진 업로드", "jpg 분석", "png 분석"],
    text: "이미지 탐색은 부유물 탐색에서 시작합니다. 분석할 JPG·PNG 같은 이미지와 학습된 YOLO PT 모델을 업로드하거나 선택한 뒤 신뢰도 기준을 정하고 분석 시작을 누르면 탐지 결과와 클래스 통계가 기록됩니다.",
    action: { label: "이미지 탐색 시작", view: "analysis" },
  },
  {
    words: ["동영상 탐색", "동영상 분석", "영상 탐색", "영상 분석", "비디오 분석", "mp4 분석", "영상 업로드"],
    text: "동영상 탐색은 부유물 탐색에서 영상 파일과 PT 모델을 선택해 실행합니다. 완료 후에는 결과 영상, 탐지 개수, 평균 신뢰도, 클래스별 통계와 처리 FPS를 탐색 기록에서 다시 볼 수 있습니다.",
    action: { label: "동영상 탐색 시작", view: "analysis" },
  },
  {
    words: ["부유물 탐색", "분석 방법", "사용 방법", "어떻게 시작", "분석 시작", "탐색 시작", "사용법"],
    text: "부유물 탐색에서는 이미지 또는 동영상과 학습된 YOLO PT 모델을 각각 업로드하거나 선택한 뒤 분석 시작 버튼을 누르면 됩니다. 완료된 결과는 탐색 기록에 자동 저장됩니다.",
    action: { label: "부유물 탐색 열기", view: "analysis" },
  },
  {
    words: ["pt", "모델 지원", "지원 모델", "yolov8", "yolo11", "yolo 11", "yolo 8"],
    text: "현재 YOLOv8 또는 YOLO11 기반 Detection·Segmentation PT 파일을 지원하며, 모델 파일은 최대 500MB까지 등록할 수 있습니다.",
    action: { label: "모델 등록하기", view: "analysis" },
  },
  {
    words: ["탐색 기록", "기록", "결과 어디", "이력", "지난 분석", "저장", "다시 보기"],
    text: "완료된 결과는 내 탐색 기록에 자동으로 저장됩니다. 분석 미디어, 탐지 결과, 클래스별 통계와 처리 지표를 다시 확인할 수 있습니다.",
    action: { label: "탐색 기록 보기", view: "records" },
  },
  {
    words: ["비교", "모델별", "어떤 모델", "성능 비교"],
    text: "탐색 기록의 상세 결과에서 네 모델의 탐지 개수, 평균 신뢰도와 처리 성능 차이를 한눈에 확인할 수 있습니다.",
    action: { label: "탐색 기록 보기", view: "records" },
  },
  {
    words: ["문의", "질문", "상담", "1 1", "일대일"],
    text: "해결되지 않은 내용은 1:1 문의로 남겨주세요. 작성한 문의와 첨부파일은 본인과 관리자만 확인할 수 있습니다.",
    action: { label: "1:1 문의하기", view: "inquiry" },
  },
];

export function FloatWatchChat({ loggedIn }: { loggedIn: boolean }) {
  const [open, setOpen] = useState(false);
  const [faqs, setFaqs] = useState<ContentItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([{ id: 1, role: "bot", text: "안녕하세요. FloatWatch 이용 방법을 안내해드릴게요." }]);
  const [input, setInput] = useState("");
  const [responding, setResponding] = useState(false);
  const [suggestions, setSuggestions] = useState(quickQuestions);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const chatRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const responseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => { api<ContentItem[]>("/content?category=faq").then(setFaqs).catch(() => {}); }, []);
  useEffect(() => {
    const update = () => setShowScrollTop(window.scrollY > 640);
    update();
    window.addEventListener("scroll", update, { passive: true });
    return () => window.removeEventListener("scroll", update);
  }, []);
  useEffect(() => { scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" }); }, [messages, open, responding]);
  useEffect(() => () => {
    if (responseTimerRef.current) clearTimeout(responseTimerRef.current);
  }, []);
  useEffect(() => {
    if (!open) return;
    const closeOnOutsidePointer = (event: PointerEvent) => {
      if (!chatRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePointer);
    return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
  }, [open]);
  const normalizedFaqs = useMemo(() => faqs.map((item) => ({ ...item, plain: stripHtml(`${item.title} ${item.content}`) })), [faqs]);

  async function ask(raw: string) {
    const question = raw.trim(); if (!question || responding) return;
    const nextId = Date.now();
    const normalized = normalize(question);
    const guide = guides.map((item) => ({ item, score: item.words.filter((word) => normalized.includes(normalize(word))).length })).sort((a, b) => b.score - a.score)[0];
    const faq = normalizedFaqs.map((item) => ({ item, score: scoreText(normalized, normalize(item.plain)) })).sort((a, b) => b.score - a.score)[0];
    let fallback: ChatMessage;
    if (faq && faq.score >= 2 && (!guide || faq.score > guide.score)) fallback = { id: nextId + 1, role: "bot", text: compactAnswer(stripHtml(faq.item.content)) };
    else if (guide && guide.score > 0) fallback = { id: nextId + 1, role: "bot", text: compactAnswer(guide.item.text), action: guide.item.action, topic: guide.item.topic };
    else fallback = { id: nextId + 1, role: "bot", text: "좋아요, 편하게 얘기해도 됩니다. 지금은 AI 연결이 원활하지 않아서 깊은 자유 대화는 어렵지만, FloatWatch 기능이나 사용 방법은 바로 안내해드릴 수 있어요.", action: { label: loggedIn ? "1:1 문의 작성" : "로그인 후 문의하기", view: loggedIn ? "inquiry" : "login" }, topic: "chat" };
    setMessages((items) => [...items, { id: nextId, role: "user", text: question }]);
    setInput("");
    setResponding(true);
    try {
      const history = messages.slice(-10).map((item) => ({ role: item.role === "bot" ? "assistant" : "user", content: item.text }));
      const result = await api<{ reply: string }>("/chat", { method: "POST", body: JSON.stringify({ message: question, history }) });
      setMessages((items) => [...items, { id: nextId + 1, role: "bot", text: result.reply }]);
      setSuggestions(quickQuestions.filter((item) => normalize(item) !== normalized).slice(0, 4));
    } catch {
      const related = followUpsByView[fallback.action?.view ?? fallback.topic ?? ""] ?? quickQuestions;
      setMessages((items) => [...items, fallback]);
      setSuggestions(related.filter((item) => normalize(item) !== normalized).slice(0, 4));
    } finally {
      setResponding(false);
    }
  }
  function submit(event: FormEvent) { event.preventDefault(); ask(input); }
  function navigate(view: string) { window.location.href = view === "login" ? "/auth?login=1" : `/auth?workspace=${view}`; }

  return <div ref={chatRef} className={`float-chat ${open ? "open" : ""}`}>
    <section className={`float-chat-panel ${open ? "is-open" : "is-closed"}`} aria-label="FloatWatch 도움말 챗봇" aria-hidden={!open}><header><div><span><Bot size={19}/></span><div><strong>FloatWatch 도우미</strong><small>AI가 서비스 이용을 안내합니다</small></div></div><button type="button" onClick={() => setOpen(false)} aria-label="챗봇 닫기"><X size={18}/></button></header><div className="float-chat-messages" ref={scrollRef}>{messages.map((message) => <article className={message.role} key={message.id}><p>{message.text}</p>{message.action && <button type="button" onClick={() => navigate(message.action!.view)}>{message.action.label}<ChevronRight size={14}/></button>}</article>)}{responding && <article className="bot float-chat-typing" aria-label="답변 작성 중"><p><span/><span/><span/></p></article>}</div>{!responding && <div className="float-chat-quick float-chat-followups">{suggestions.map((question) => <button type="button" onClick={() => ask(question)} key={question}>{question}</button>)}</div>}<form onSubmit={submit}><input value={input} onChange={(event) => setInput(event.target.value)} maxLength={300} placeholder={responding ? "답변을 준비하고 있습니다" : "궁금한 내용을 입력하세요"} aria-label="챗봇 질문" disabled={responding}/><button type="submit" disabled={!input.trim() || responding} aria-label="질문 보내기"><Send size={17}/></button></form><footer>OpenAI 기반 답변 · 중요한 정보는 다시 확인해 주세요.</footer></section>
    {showScrollTop && !open && <button className="float-scroll-top" type="button" onClick={() => window.scrollTo({ top: 0, left: 0, behavior: "smooth" })} aria-label="맨 위로 이동" title="맨 위로"><ArrowUp size={20}/></button>}
    <button className="float-chat-toggle" type="button" onClick={() => setOpen((value) => !value)} aria-label={open ? "챗봇 닫기" : "도움말 챗봇 열기"}>{open ? <X size={25}/> : <Bot size={30}/>}</button>
  </div>;
}

function normalize(value: string) { return value.toLowerCase().replace(/[^0-9a-z가-힣]+/g, " ").trim(); }
function stripHtml(value: string) { return value.replace(/<br\s*\/?>/gi, " ").replace(/<[^>]*>/g, " ").replace(/&nbsp;/g, " ").replace(/\s+/g, " ").trim(); }
function compactAnswer(value: string) {
  const text = stripHtml(value);
  const sentences = text.match(/[^.!?。]+[.!?。]?/g)?.map((item) => item.trim()).filter(Boolean) ?? [text];
  const concise = sentences.slice(0, 2).join(" ");
  if (concise.length <= 145) return concise;
  const shortened = concise.slice(0, 142);
  const boundary = shortened.lastIndexOf(" ");
  return `${shortened.slice(0, boundary > 90 ? boundary : 142).replace(/[.,!?]?$/, "")}…`;
}
function scoreText(question: string, target: string) { const tokens = question.split(" ").filter((token) => token.length > 1); return tokens.reduce((score, token) => score + (target.includes(token) ? 1 : 0), 0); }
