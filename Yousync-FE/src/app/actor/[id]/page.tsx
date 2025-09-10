"use client"
import React, { useCallback, useEffect, useState } from 'react'
import { useParams } from 'next/navigation';
import axios from 'axios';
import { ActorMovie } from '@/types/actor';
import { ActorContainer } from '@/components/actor/ActorContainer';

export default function ActorPage() {
    const params = useParams<{id: string}>();
    const actorName = decodeURIComponent(params.id);
    const [movies, setMovies] = useState<ActorMovie[]>([]);
    const [isLoading, setIsLoading] = useState(false);

    // getActorMovies 함수로 빼기
    const getActorMovies = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await axios.get<ActorMovie[]>(`${process.env.NEXT_PUBLIC_API_BASE_URL}/actors/${actorName}`)
            console.log(res.data)
            setMovies(res.data);
        } catch (error) {
            console.error("영화목록 가져오기 실패", error);
        } finally {
            setIsLoading(false);
        }
    }, [actorName])

    useEffect(() => {
        if(actorName) {
            getActorMovies();
        }
    }, [actorName])

    return (
        <ActorContainer 
            movies={movies} 
            actorName={actorName} 
            isLoading={isLoading} 
        />
    );
}