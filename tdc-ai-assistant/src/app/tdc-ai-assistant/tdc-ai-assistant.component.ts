// src/app/tdc-ai-assistant/tdc-ai-assistant.component.ts
import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import {MarkdownToHtmlPipe} from '../pipes/markdown-to-html.pipe';

interface AskRequest {
  text: string;
  limit?: number;
}

interface AskResponse {
  answer: string;
  sources: string[];
  time_taken: number;
  timings?: {
    embedding_ms?: number;
    qdrant_ms?: number;
    mongo_ms?: number;
    gpt_ms?: number;
  };
}

interface DebugRequest {
  text: string;
  limit?: number;
}

interface DebugHit {
  mongo_id: string;
  title?: string | null;
  type?: string | null;
  vector_type?: string | null;
  score: number;
}

interface DebugResponse {
  query: string;
  limit: number;
  hits: DebugHit[];
}

@Component({
  selector: 'app-tdc-ai-assistant',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule, MarkdownToHtmlPipe],
  templateUrl: './tdc-ai-assistant.component.html',
  styleUrls: ['./tdc-ai-assistant.component.scss'],
})
export class TdcAiAssistantComponent {
  // Ajuste aqui para apontar para seu search_service
  private readonly apiBaseUrl = 'http://localhost:8000';

  question = '';
  isLoading = false;
  isDebugLoading = false;
  errorMessage: string | null = null;

  response: AskResponse | null = null;
  debugData: DebugResponse | null = null;

  showDebugPanel = false;

  suggestedQuestions: string[] = [
    'Rodrigo Tavares estará no TDC Porto Alegre?',
    'Quais trilhas de IA e dados vão ter em Porto Alegre?',
    'Quando acontece o TDC Experience Porto Alegre 2025?',
    'Quais atividades têm foco em RAG e LLM?',
  ];

  constructor(private http: HttpClient) {}

  ask(): void {
    const text = this.question?.trim();
    if (!text) {
      this.errorMessage = 'Digite uma pergunta para o assistente.';
      return;
    }

    this.isLoading = true;
    this.errorMessage = null;
    this.response = null;

    const payload: AskRequest = {
      text,
      limit: 5,
    };

    this.http.post<AskResponse>(`${this.apiBaseUrl}/ask`, payload).subscribe({
      next: (res) => {
        this.response = res;
        this.isLoading = false;
      },
      error: (err) => {
        this.isLoading = false;
        this.errorMessage =
          err?.error?.detail ||
          'Ocorreu um erro ao consultar o assistente. Verifique a API e tente novamente.';
      },
    });
  }

  runDebug(): void {
    const text = this.question?.trim();
    if (!text) {
      this.errorMessage =
        'Para ver os bastidores da busca, primeiro digite a pergunta.';
      return;
    }

    this.isDebugLoading = true;
    this.errorMessage = null;
    this.debugData = null;

    const payload: DebugRequest = {
      text,
      limit: 6,
    };

    this.http
      .post<DebugResponse>(`${this.apiBaseUrl}/debug/search`, payload)
      .subscribe({
        next: (res) => {
          this.debugData = res;
          this.isDebugLoading = false;
          this.showDebugPanel = true;
        },
        error: (err) => {
          this.isDebugLoading = false;
          this.errorMessage =
            err?.error?.detail ||
            'Ocorreu um erro ao consultar o endpoint de debug do Qdrant.';
        },
      });
  }

  useSuggestion(text: string): void {
    this.question = text;
    this.ask();
  }

  get totalTimeMs(): string {
    if (!this.response?.time_taken) return '-';
    return `${(this.response.time_taken * 1000).toFixed(0)} ms`;
  }

  getTimingLabel(key: keyof NonNullable<AskResponse['timings']>): string {
    if (!this.response?.timings) return '-';
    const value = this.response.timings[key];
    if (value == null) return '-';
    return `${value.toFixed(1)} ms`;
  }

  getHitBadgeLabel(hit: DebugHit): string {
    if (!hit.type) return 'IDX';
    if (hit.type === 'event_info') return 'EVENTO';
    if (hit.type === 'talk') {
      if (hit.vector_type === 'person') return 'PALESTRANTE';
      if (hit.vector_type === 'topic') return 'TEMA';
      return 'TALK';
    }
    return hit.type.toUpperCase();
  }

  trackByMongoId(_index: number, hit: DebugHit): string {
    return hit.mongo_id;
  }
}
